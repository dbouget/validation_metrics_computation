import multiprocessing
import itertools
import logging
import os.path
import traceback
import sqlite3
import csv
import pandas as pd
from math import ceil
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from ..Computation.dice_computation_instance import separate_dice_computation
from ..Validation.instance_segmentation_validation import *
from ..Utils.resources import SharedResources
from ..Utils.PatientMetricsStructure import PatientMetrics
from ..Utils.io_converters import get_fold_from_file, open_image_file, save_image_file, is_valid_extension
from ..Validation.validation_utilities import best_segmentation_probability_threshold_analysis, compute_fold_average
from ..Validation.extra_metrics_computation import compute_patient_extra_metrics


class ModelValidation:
    """
    Compute performances metrics after k-fold cross-validation from sets of inference.
    The results will be stored inside a Validation sub-folder placed within the provided destination directory.
    """
    def __init__(self):
        self.data_root = SharedResources.getInstance().data_root
        self.input_folder = SharedResources.getInstance().validation_input_folder
        base_output_folder = SharedResources.getInstance().validation_output_folder

        if base_output_folder is not None and base_output_folder != "":
            self.output_folder = os.path.join(base_output_folder, 'Validation')
        else:
            self.output_folder = os.path.join(self.input_folder, 'Validation')
        os.makedirs(self.output_folder, exist_ok=True)

        self.fold_number = SharedResources.getInstance().validation_nb_folds
        self.split_way = SharedResources.getInstance().validation_split_way
        self.metric_names = []
        for m in SharedResources.getInstance().validation_metric_names:
            self.metric_names.extend([f'PiW {m}', f'OW {m}'])
        self.detection_overlap_thresholds = SharedResources.getInstance().validation_detection_overlap_thresholds
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix
        self.patients_metrics = {}

    def run(self):
        logging.info("Computing metrics for cohort.")
        self.__compute_metrics()
        logging.info("Running optimal thresholds analysis.")
        class_optimal = best_segmentation_probability_threshold_analysis(self.output_folder,
                                                                         detection_overlap_thresholds=self.detection_overlap_thresholds)
        if len(SharedResources.getInstance().validation_metric_names) != 0:
            logging.info("Computing extra metrics for cohort.")
            self.__compute_extra_metrics(class_optimal=class_optimal)
            logging.info("Re-exporting results to CSV with extra metrics...")
            self.__sqlite_to_csv("total_results", self.dice_output_filename)
            for c in SharedResources.getInstance().validation_class_names:
                self.__sqlite_to_csv(f"class_{c}", self.class_dice_output_filenames[c])
                
        # Read all extra metric columns actually present in the CSV, not just the ones from the current config.        
        tmp = pd.read_csv(self.dice_output_filename)
        all_extra_metric_names = [col for col in tmp.columns if col not in self.results_df_fixed_columns]
            
        logging.info("Computing average metrics for the cohort.")
        # All
        compute_fold_average(self.output_folder, class_optimal=class_optimal, metrics=all_extra_metric_names, condition='All')
        # Positive, based on given ground truth volume limit
        compute_fold_average(self.output_folder, class_optimal=class_optimal, metrics=all_extra_metric_names, condition='Positive')
        # True positive, based on given detection_overlap_thresholds
        compute_fold_average(self.output_folder, class_optimal=class_optimal, metrics=all_extra_metric_names, condition='TP')

        self.conn.close()

    def __compute_metrics(self):
        """
        Generate the Dice scores (and default instance detection metrics) for all the patients and 10 probability
        thresholds equally-spaced. All the computed results will be stored inside all_dice_scores.csv.
        The results are saved after each patient, making it possible to resume the computation if a crash occurred.
        @TODO. Include an override flag to recompute anyway.
        :return:
        """
        cross_validation_description_file = os.path.join(self.input_folder, 'cross_validation_folds.txt')
        self.dice_output_filename = os.path.join(self.output_folder, 'all_dice_scores.csv')
        self.class_dice_output_filenames = {}
        for c in SharedResources.getInstance().validation_class_names:
            self.class_dice_output_filenames[c] = os.path.join(self.output_folder, c + '_dice_scores.csv')

        # Define the column schema shared by all result tables
        self.results_df_fixed_columns = ['Fold', 'Patient', 'Threshold']
        self.results_df_fixed_columns.extend(["PiW Dice", "PiW Recall", "PiW Precision", "PiW F1"])
        # self.results_df_base_columns.extend(["PaW Dice", "PaW Recall", "PaW Precision", "PaW F1"])
        self.results_df_fixed_columns.extend(["GT volume (ml)", "True Positive", "Detection volume (ml)"])
        self.results_df_fixed_columns.extend(["OW Global Recall", "OW Global Precision", "OW Global F1", "OW Dice",
                                             "OW Dice (std)", "OW Recall", "OW Recall (std)", "OW Precision",
                                             "OW Precision (std)", "OW F1", "OW F1 (std)", '#GT', '#Det'])
        self.results_df_base_columns = self.results_df_fixed_columns + self.metric_names

        # Connect to SQLite database
        # WAL mode ensures crash-safe writes
        self.db_path = os.path.join(self.output_folder, "results.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        cursor = self.conn.cursor()

        # Initialize or resume the total_results table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='total_results'")
        if cursor.fetchone() is None:
            # First run: seed from existing CSV if available, otherwise start empty
            if os.path.exists(self.dice_output_filename):
                results_df = pd.read_csv(self.dice_output_filename)
                if results_df.columns[0] != 'Fold':
                    results_df = pd.read_csv(self.dice_output_filename, index_col=0)
                missing_metrics = [x for x in self.metric_names if
                                not x in list(results_df.columns)[1:]]
                for m in missing_metrics:
                    results_df[m] = None
            else:
                results_df = pd.DataFrame(columns=self.results_df_base_columns)
                
            results_df.to_sql("total_results", self.conn, if_exists="replace", index=False)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_total_search ON total_results ([Patient], [Fold], [Threshold]);")
            self.conn.commit()
        else:
            logging.info("Existing database found, resuming from SQL...")
            if not os.path.exists(self.dice_output_filename):
                self.__sqlite_to_csv("total_results", self.dice_output_filename)

        # Initialize or resume per-class tables
        for c in SharedResources.getInstance().validation_class_names:
            table_name = f"class_{c}"
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")

            if cursor.fetchone() is None:
                if not os.path.exists(self.class_dice_output_filenames[c]):
                    class_df = pd.DataFrame(columns=self.results_df_base_columns)
                else:
                    class_df = pd.read_csv(self.class_dice_output_filenames[c])
                    if class_df.columns[0] != 'Fold':
                        class_df = pd.read_csv(self.class_dice_output_filenames[c], index_col=0)
                    missing_metrics = [x for x in self.metric_names if
                                    x not in list(class_df.columns)[1:]]
                    for m in missing_metrics:
                        class_df[m] = None
                    
                class_df.to_sql(table_name, self.conn, if_exists="replace", index=False)
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_class_{c} ON [{table_name}] ([Patient], [Fold], [Threshold]);")
                self.conn.commit()
            else:
                logging.info(f"Existing table found for class {c}, resuming from SQL...")
                if not os.path.exists(self.class_dice_output_filenames[c]):
                    self.__sqlite_to_csv(f"class_{c}", self.class_dice_output_filenames[c])

        # Add any missing extra metric columns to all tables (e.g. when new metrics are added after initial run)
        for metric_name in self.metric_names:
            for table in ["total_results"] + [f"class_{c}" for c in SharedResources.getInstance().validation_class_names]:
                try:
                    cursor.execute(f"ALTER TABLE [{table}] ADD COLUMN [{metric_name}] REAL")
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass  # Column already exists

        # Process each fold
        results_per_folds = []
        for fold in range(0, self.fold_number):
            logging.info(f'\nProcessing fold {fold+1}/{self.fold_number}.\n')
            if self.split_way == 'two-way':
                test_set, _ = get_fold_from_file(filename=cross_validation_description_file, fold_number=fold)
            else:
                val_set, test_set = get_fold_from_file(filename=cross_validation_description_file, fold_number=fold)
            results = self.__compute_metrics_for_fold(data_list=test_set, fold_number=fold)
            results_per_folds.append(results)

        logging.info("Exporting results to CSV...")
        self.__sqlite_to_csv("total_results", self.dice_output_filename)
        for c in SharedResources.getInstance().validation_class_names:
            self.__sqlite_to_csv(f"class_{c}", self.class_dice_output_filenames[c])

    def __compute_metrics_for_fold(self, data_list, fold_number):
        if not os.path.exists(os.path.join(SharedResources.getInstance().validation_input_folder, "predictions", str(fold_number))):
            logging.warning(f"No predictions folder for fold {fold_number} -- Skipping!")
            print(f"No predictions folder for fold {fold_number} -- Skipping!")
            return 0

        for i, patient in enumerate(tqdm(data_list)):
            uid = None
            try:
                # Option1. Working for files using the original naming conventions.
                if SharedResources.getInstance().validation_use_index_naming_convention:
                    pid = patient.split('_')[1]
                    sub_folder_index = str(ceil(int(pid) / 200))  # patient.split('_')[0]
                    pid = pid + '_' + patient.split('_')[4]
                else:
                    # Option2. For files not following the original naming conventions
                    pid = patient
                    sub_folder_index = None

                uid = str(fold_number) + '_' + pid
                # Placeholder for holding all metrics for the current patient
                patient_metrics = PatientMetrics(id=uid, patient_id=pid, fold_number=fold_number,
                                                 class_names=SharedResources.getInstance().validation_class_names)
                
                # Load any previously computed metrics from the database
                patient_metrics.init_from_db(self.conn)
                success = self.__identify_patient_files(patient_metrics, sub_folder_index, fold_number)
                self.patients_metrics[uid] = patient_metrics
                
                # Checking if values have already been computed for the current patient to skip it if so.
                if patient_metrics.is_complete():
                    continue
                if not success:
                    print('Input files not found for patient {}\n'.format(uid))
                    continue
                
                self.__generate_dice_scores_for_patient(patient_metrics, fold_number)
            except Exception as e:
                print('Issue processing patient {}\n'.format(uid))
                print(traceback.format_exc())
                continue
        return 0

    def __identify_patient_files(self, patient_metrics: PatientMetrics, folder_index: int, fold_number: int) -> bool:
        """
        Asserts the existence of the raw files on disk for computing the metrics for the current patient.

        Parameters
        ----------
        patient_metrics: PatientMetrics
            Object holding all computed metrics for the current patient
        folder_index: int
            Index value for the folder on disk to investigate
        fold_number: int
            Current fold number for looking into the correct folder on disk

        Returns
        ----------
        bool
             Boolean indicating whether all patient files were correctly identified or not.
        """
        use_internal_convention = SharedResources.getInstance().validation_use_index_naming_convention
        uid = patient_metrics.patient_id
        if use_internal_convention:
            uid = patient_metrics.patient_id.split('_')[0]
        classes = SharedResources.getInstance().validation_class_names
        nb_classes = len(classes)
        patient_filenames = {}

        # Iterating over all classes, where independent files are expected
        for c in range(nb_classes):
            patient_filenames[classes[c]] = []
            gt_suffix = self.gt_files_suffix[c]
            pred_suffix = self.prediction_files_suffix[c]

            # Annoying, but independent of extension
            # @TODO. must load images with SimpleITK to be completely generic.
            detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number), uid)
            if folder_index is not None:
               detection_image_base = os.path.join(self.input_folder, 'predictions', str(fold_number),
                                                   folder_index + '_' + uid)

            detection_filename = None
            for _, _, files in os.walk(detection_image_base):
                for f in files:
                    if pred_suffix in f and is_valid_extension(fn=f, extensions=SharedResources.getInstance().valid_file_extensions):
                        if use_internal_convention and patient_metrics.patient_id.split('_')[1] in f.split('_'):
                            detection_filename = os.path.join(detection_image_base, f)
                        elif not use_internal_convention:
                            detection_filename = os.path.join(detection_image_base, f)
                break
            if not os.path.exists(detection_filename):
                print("No detection file found for class {} in patient {}".format(c, patient_metrics.unique_id))
                return False

            # Identification of the ground truth filename
            patient_extended = uid
            if folder_index is not None:
                patient_extended = os.path.basename(detection_filename).split(pred_suffix)[0][:-1]

            ground_truth_base = os.path.join(self.data_root, uid, patient_extended)
            if SharedResources.getInstance().validation_use_index_naming_convention and folder_index is not None:
                ground_truth_base = os.path.join(self.data_root, folder_index, uid, 'segmentations', patient_extended)

            ground_truth_filename = None
            for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                for f in files:
                    if os.path.basename(ground_truth_base) in f and gt_suffix in f and is_valid_extension(fn=f, extensions=SharedResources.getInstance().valid_file_extensions):
                        ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                break
            # The ground truth filename inside the folder does not match the folder name, looking for the first eligible file if any
            if ground_truth_filename is None:
                for _, _, files in os.walk(os.path.dirname(ground_truth_base)):
                    for f in files:
                        if gt_suffix in f and is_valid_extension(fn=f, extensions=SharedResources.getInstance().valid_file_extensions):
                            ground_truth_filename = os.path.join(os.path.dirname(ground_truth_base), f)
                    break

            # Specific actions for remapping BraTS results to match the whole tumor and tumor core categories
            if SharedResources.getInstance().validation_use_brats_data and (classes[c] == 'whole' or classes[c] == 'core'):
                detection_ni = nib.load(detection_filename)
                ground_truth_filename = os.path.join(detection_image_base, patient_extended + '_' + gt_suffix)
            # The ground truth for the BraTS images is stored a bit differently
            elif SharedResources.getInstance().validation_use_brats_data and classes[c] == 'tumor':
                detection_ni = nib.load(detection_filename)
                raw_gt = nib.load(ground_truth_filename).get_fdata()[:]
                ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_tumor.nii.gz")
                if not os.path.exists(ground_truth_filename):
                    new_gt = np.zeros(detection_ni.get_fdata().shape)
                    new_gt[raw_gt == 1] = 1
                    nib.save(nib.Nifti1Image(new_gt, detection_ni.affine), ground_truth_filename)
                tmp_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_necrosis.nii.gz")
                if not os.path.exists(tmp_filename):
                    new_gt = np.zeros(detection_ni.get_fdata().shape)
                    new_gt[raw_gt == 2] = 1
                    new_gt[raw_gt == 1] = 0
                    nib.save(nib.Nifti1Image(new_gt, detection_ni.affine), tmp_filename)
            elif SharedResources.getInstance().validation_use_brats_data and classes[c] == 'necrosis':
                ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_necrosis.nii.gz")

            # detection_ni = nib.load(detection_filename)
            # # If there's no ground truth, we assume the class to be empty for this patient and create an
            # # empty ground truth volume.
            # if ground_truth_filename is None or not os.path.exists(ground_truth_filename):
            #     ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_" + classes[c] + ".nii.gz")
            #     if not os.path.exists(ground_truth_filename):
            #         empty_gt = np.zeros(detection_ni.get_fdata().shape)
            #         nib.save(nib.Nifti1Image(empty_gt, detection_ni.affine), ground_truth_filename)
            # else:
            #     file_stats = os.stat(detection_filename)
            #     ground_truth_ni = nib.load(ground_truth_filename)
            #     if len(ground_truth_ni.shape) == 4:
            #         ground_truth_ni = nib.four_to_three(ground_truth_ni)[0]
            #
            #     if file_stats.st_size == 0:
            #         nib.save(nib.Nifti1Image(np.zeros(ground_truth_ni.get_shape), affine=ground_truth_ni.affine),
            #                  detection_filename)
            #
            #     if detection_ni.shape != ground_truth_ni.shape:
            #         return False
            # If there's no ground truth, we assume the class to be empty for this patient and create an
            # empty ground truth volume.
            if ground_truth_filename is None or not os.path.exists(ground_truth_filename):
                detection_array, file_extension, input_spec = open_image_file(detection_filename)
                ground_truth_filename = os.path.join(os.path.dirname(detection_filename), uid + "_groundtruth_" +
                                                     classes[c] + file_extension)
                if not os.path.exists(ground_truth_filename):
                    empty_gt = np.zeros(detection_array.shape)
                    save_image_file(empty_gt, ground_truth_filename, specifics=input_spec)
            else:
                file_stats = os.stat(detection_filename)

                if file_stats.st_size == 0:
                    ground_truth_array, _, ground_truth_input_spec = open_image_file(ground_truth_filename)
                    detection_array, file_extension, input_spec = open_image_file(detection_filename)
                    save_image_file(np.zeros(shape=ground_truth_array.shape), detection_filename,
                                    specifics=ground_truth_input_spec)

                # if detection_array.shape != ground_truth_array.shape:
                #     return False

            patient_filenames[classes[c]] = [ground_truth_filename, detection_filename]
        patient_metrics.set_patient_filenames(patient_filenames)
        return True

    def __generate_dice_scores_for_patient(self, patient_metrics, fold_number):
        """
        Compute the basic metrics for all classes of the current patient
        :return:
        """
        uid = patient_metrics.patient_id
        classes = SharedResources.getInstance().validation_class_names
        nb_classes = len(classes)
        thr_range = np.arange(0.1, 1.1, 0.1)
        cursor = self.conn.cursor()

        # Iterating over all classes, where independent files are expected
        for c in range(nb_classes):
            gt_filename, det_filename = patient_metrics.get_class_filenames(c)
            gt, _, gt_specs = open_image_file(gt_filename)
            detection, _, det_specs = open_image_file(det_filename)
            gt[gt >= 1] = 1

            class_tp_threshold = SharedResources.getInstance().validation_true_positive_volume_thresholds[c]
            gt_volume = np.count_nonzero(gt) * np.prod(det_specs[1]) * 1e-3
            tp_state = True if gt_volume > class_tp_threshold else False
            extra = [np.round(gt_volume, 4), tp_state, det_specs[1]]
            pat_results = []

            # Compute Dice scores across all thresholds, using multiprocessing if configured
            if SharedResources.getInstance().number_processes > 1:
                pool = multiprocessing.Pool(processes=SharedResources.getInstance().number_processes)
                pat_results = pool.map(separate_dice_computation, zip(thr_range,
                                                                      itertools.repeat(fold_number),
                                                                      itertools.repeat(gt),
                                                                      itertools.repeat(detection),
                                                                      itertools.repeat(uid),
                                                                      itertools.repeat(extra)
                                                                      )
                                       )
                pool.close()
                pool.join()
            else:
                for thr_value in thr_range:
                    thr_res = separate_dice_computation([thr_value, fold_number, gt, detection, uid, extra])
                    pat_results.append(thr_res)

            patient_metrics.set_class_regular_metrics(classes[c], pat_results)

            # Write per-class results to the database
            table_name = f"class_{classes[c]}"
            columns = list(self.results_df_base_columns)

            for ind, th in enumerate(thr_range):
                th = np.round(th, 2)

                query = f"SELECT * from [{table_name}] where [Patient] = ? AND [Fold] = ? AND [Threshold] = ?"
                cursor.execute(query, (str(uid), fold_number, th))
                row = cursor.fetchone()

                pat_res_tmp =[float(x) if isinstance(x, np.float32) else x for x in pat_results[ind][0]]
                if row is None:
                    extra_metrics = [None] * 2 * len(SharedResources.getInstance().validation_metric_names)
                    ind_values = np.asarray(pat_res_tmp + extra_metrics)

                    column_names_str = ", ".join([f"[{col}]" for col in columns])
                    placeholders = ", ".join(["?"] * len(columns))

                    insert_query = f"INSERT INTO {table_name} ({column_names_str}) VALUES ({placeholders})"
                    cursor.execute(insert_query, tuple(ind_values))

                else:
                    ind_values = pat_res_tmp + list(row[len(pat_results[ind][0]):])
                    set_clause = ", ".join([f"[{col}] = ?" for col in columns])
                    update_query = f"UPDATE {table_name} SET {set_clause} WHERE [Patient] = ? AND [Fold] = ? AND [Threshold] = ?"
                    cursor.execute(update_query, tuple(ind_values) + (uid, fold_number, th))
                    
            self.conn.commit()
            

        # Should compute the class macro-average results if multiple classes
        class_averaged_results = None
        class_results = []
        for c in classes:
            pat_class_results = patient_metrics.get_class_metrics(c)
            pat_class_extra_metrics = patient_metrics.get_class_extra_metrics_without_header(c)
            final_pat_class_res = [pat_class_results[x] for x in range(len(thr_range))]
            if len(SharedResources.getInstance().validation_metric_names) != 0:
                final_pat_class_res = [pat_class_results[x] + pat_class_extra_metrics[x] for x in range(len(thr_range))]
            class_results.append(final_pat_class_res)
        class_averaged_results = np.average(np.asarray(class_results).astype('float32')[:, :, 1:], axis=0).astype(float)
        current_columns = pd.read_sql_query(f"SELECT * FROM {'total_results'} LIMIT 0", self.conn).columns.tolist()

        for ind, th in enumerate(thr_range):
            th = np.round(th, 2)

            query = f"SELECT * from {'total_results'} where [Patient] = ? AND [Fold] = ? AND [Threshold] = ?"
            cursor.execute(query, (str(uid), fold_number, th))

            row = cursor.fetchone()

            if row is None:
                ind_values = [fold_number, uid, np.round(th, 2)] + list(class_averaged_results[ind])
                # Pad with None if fewer values than columns (e.g. when extra metrics are missing)
                if len(ind_values) < len(current_columns):
                    ind_values += [None] * (len(current_columns) - len(ind_values))

                column_names_str = ", ".join([f"[{col}]" for col in current_columns])
                placeholders = ", ".join(["?"] * len(current_columns))
                insert_query = f"INSERT INTO {'total_results'} ({column_names_str}) VALUES ({placeholders})"
                cursor.execute(insert_query, tuple(ind_values))
            else:
                row_dict = dict(zip(current_columns, row))
                new_values = [fold_number, uid, th] + list(class_averaged_results[ind])

                for i, val in enumerate(new_values):
                    if i < len(current_columns):
                        row_dict[current_columns[i]] = float(val) if isinstance(val, np.float32) else val

                ind_values = [row_dict.get(col, None) for col in current_columns]
                set_clause = ", ".join([f"[{col}] = ?" for col in current_columns])

                update_query = f"UPDATE {'total_results'} SET {set_clause} WHERE [Patient] = ? AND [Fold] = ? AND [Threshold] = ?"
                cursor.execute(update_query, tuple(ind_values) + (str(uid), fold_number, np.round(th, 2)))

        self.conn.commit()
    
    def __compute_extra_metrics(self, class_optimal: dict = {}):
        """
        Computes additional metrics at the optimal threshold for each class.
        Results are written to both the per-class table and total_results in SQLite.
        @TODO. Would need to properly compute metrics average over the different classes to fill in the total_results table (or just skip it).
        """
        classes = SharedResources.getInstance().validation_class_names
        for c in classes:
            optimal_values = class_optimal[c]['All']
            if len(SharedResources.getInstance().validation_metric_names) < 10:
                batch_results =[] 
                dump = 0
                with ThreadPoolExecutor(max_workers=SharedResources.getInstance().number_processes) as executor:
                    futures = {executor.submit(partial(self.__patient_metrics_computation, c=c, classes=classes, 
                                                       optimal_values=optimal_values), item): item for item in self.patients_metrics}
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        _ = futures[future]
                        try:
                            results = future.result()
                            batch_results.append(results)
                            dump += 1
                            if dump % SharedResources.getInstance().number_processes == 0:
                                self.__update_database(batch_results)
                                batch_results.clear()
                        except Exception as e:
                            continue
            else:
                original_processes = SharedResources.getInstance().number_processes
                SharedResources.getInstance().number_processes = 10
                for p in tqdm(self.patients_metrics):
                    recomputation = False
                    try:
                        # Initializing/completing the list which will hold the extra metrics
                        self.patients_metrics[p].setup_extra_metrics(self.metric_names)
                        pat_metrics, recomputation = compute_patient_extra_metrics(self.patients_metrics[p], classes.index(c), optimal_values[1],
                                                                    SharedResources.getInstance().validation_metric_names)
                        if recomputation:
                            self.patients_metrics[p].set_optimal_class_extra_metrics(classes.index(c), optimal_values[1], pat_metrics)
                            cursor = self.conn.cursor()
                            for pm in pat_metrics:
                                metric_name = pm[0]
                                metric_value = pm[1]

                                thr_to_match = float(np.round(optimal_values[1], 2))

                                update_query = f"UPDATE [class_{c}] SET [{metric_name}] = ? WHERE [Patient] = ? AND [Threshold] = ?"
                                cursor.execute(update_query, (metric_value, str(self.patients_metrics[p].patient_id), thr_to_match))
                            
                                update_query = f"UPDATE total_results SET [{metric_name}] = ? WHERE [Patient] = ? AND [Threshold] = ?"                        
                                cursor.execute(update_query, (metric_value, str(self.patients_metrics[p].patient_id), thr_to_match))
                    except Exception as e:
                        logging.error(f"Computing extra metrics for patient {self.patients_metrics[p].patient_id} failed with: {e}.\n{traceback.format_exc()}")
                        continue
                    finally:
                        if recomputation:
                            self.conn.commit()
                SharedResources.getInstance().number_processes = original_processes


    def __patient_metrics_computation(self, p, c, classes, optimal_values):
        recomputation = False
        result = None
        try:
            # Initializing/completing the list which will hold the extra metrics
            thr_to_match = float(np.round(optimal_values[1], 2))
            self.patients_metrics[p].setup_extra_metrics(self.metric_names)
            pat_metrics, recomputation = compute_patient_extra_metrics(self.patients_metrics[p], classes.index(c), optimal_values[1],
                                                        SharedResources.getInstance().validation_metric_names)
            if recomputation:
                self.patients_metrics[p].set_optimal_class_extra_metrics(classes.index(c), optimal_values[1], pat_metrics)

            result = [p, c, thr_to_match, pat_metrics, recomputation] 
        except Exception as e:
            logging.error(f"Computing extra metrics for patient {self.patients_metrics[p].patient_id} failed with: {e}.\n{traceback.format_exc()}")
        finally:
            return result

    def __update_database(self, batch_results):
        try:
            cursor = self.conn.cursor()
            for results in batch_results:
                p = results[0]
                c = results[1]
                thr_to_match  = results[2]
                pat_metrics = results[3]
                recompute = results[4]
                if recompute:
                    for pm in pat_metrics:
                        metric_name = pm[0]
                        metric_value = pm[1]

                        update_query = f"UPDATE [class_{c}] SET [{metric_name}] = ? WHERE [Patient] = ? AND [Threshold] = ?"
                        cursor.execute(update_query, (metric_value, str(self.patients_metrics[p].patient_id), thr_to_match))
                    
                        update_query = f"UPDATE total_results SET [{metric_name}] = ? WHERE [Patient] = ? AND [Threshold] = ?"                        
                        cursor.execute(update_query, (metric_value, str(self.patients_metrics[p].patient_id), thr_to_match))
                self.conn.commit()
        except Exception as e:
            logging.error(f"Commiting to the database after parallel metrics computation failed with {e}")

    def __sqlite_to_csv(self, table_name, csv_path):
        """
        Exports a full SQLite table to a CSV file using a streaming cursor
        to avoid loading the entire table into memory.
        """
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM [{table_name}]")
        headers = [description[0] for description in cursor.description]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(cursor)