from libemg._datasets.dataset import Dataset
from libemg.data_handler import OfflineDataHandler, RegexFilter
import os

class FullReachableWorkspace(Dataset):
	"""Biologically-inspired transhumeral targeted muscle reinnervation dataset with data gathered in limb positions covering the
		full human reachable workspace. 

	Notes from Laura:
		7 sensors placed on each arm, set braced to True or False (default)
			Dominant arm was placed in a restrictive brace to elicit isometric muscle contractions
		
		11 limb position labels, but only 9 limb positions because the two cross body postions were performed individually
			lp07 (unbraced arm) and lp08 (braced arm) are the same limb position
			lp09 (unbraced arm) and lp10 (braced arm) are the same limb position
			All other limb positions were carried out with both arms at the same time

		Delsys Trigno Base Station and sensors were used for the data collection
			Sampling rates:
				EMG ~1111 Hz
				IMU Acc & Gyr ~148 Hz
				IMU Mag ~74 Hz
		
		All of the datasheets have had a bandpass filter and notch filter passed over the raw EMG data

	Limb position variation:
		Use all the limb positions below shoulder height as training (except behind the back)
			lp00, lp01, lp03, lp04, lp07
		Use all the limb positions above shoulder height and behind the back as test set
			lp02, lp05, lp06, lp09
	"""

	def __init__(self, dataset_folder="FullReachableWorkspace", braced=False, training_limb_positions=0, train_participants=[], test_participants=[], neutral_transitions=False, oitmr=False):
		Dataset.__init__(self, 
						1111, # sampling rate
						7, # number of sensors (note we are using only data from the unbraced arm, set braced=True to use braced arm data)
						'Delsys',
						16, # number of participants
						{0: "Neutral", 1: "Elbow Flexion", 2: "Elbow Extension", 3: "Elbow Extension to Neutral", 
						 4: "Pronation", 5: "Supination", 6: "Supination to Neutral", 
						 7: "Wrist Flexion", 8: "Wrist Extension", 9: "Wrist Extension to Neutral", 
						 10: "Radial Deviation", 11: "Ulnar Deviation", 12: "Ulnar Deviation to Neutral",
						 13: "Hand Close", 14: "Hand Open"}, 
						'16 participants of 3 trials each with 9 limb positions in each trial (27 reps total per participant)',
						"The Full Reachable Workspace Dataset contains 15 classes in 9 limb positions. It was created with the intent of training models for transhumeral targeted muscle reinnervation prosthetic control.",
						"https://ieeexplore.ieee.org/document/8630679")
		self.url = "https://conferences.lib.unb.ca/index.php/mec/article/view/2479"
		self.dataset_folder = dataset_folder
		self.braced = braced
		self.training_limb_positions = training_limb_positions
		self.train_participants = train_participants
		self.test_participants = test_participants
		self.neutral_transitions = neutral_transitions
		self.oitmr = oitmr 


	def prepare_data(self, split = False, subjects_values = None, sets_values = None, reps_values = None,
					 classes_values = None):

		print('\nPlease cite: ' + self.citation+'\n')
		if (not self.check_exists(self.dataset_folder)):
			# self.download(self.url, self.dataset_folder)
			print("Please contact Laura Petrich about getting the Full Reachable Workspace Dataset: laurapetrich@ualberta.ca") 
			return 

		if subjects_values is None:
			subjects_values = [f"{i:02}" for i in range(1, self.num_subjects + 1)]
		
		if sets_values is None:
			sets_values = ['trial1', 'trial2', 'trial3']
		
		if reps_values is None:
			# reps_values = [f"{i:02}" for i in range(11)] # reps 0 - 10; corresponds with 11 limb positions carried out in each trial
			if self.braced == False:
				reps_values = ['00', '01', '02', '03', '04', '05', '06', '07', '09'] # isotonic muscle contractions (unbraced arm)
			else:
				reps_values = ['00', '01', '02', '03', '04', '05', '06', '08', '10'] # isometric muscle contractions (braced arm)				
		if classes_values is None:
			if self.oitmr:
				# only use the classes of interest for our OI TMR control study experiment
					classes_values = ['00', '01', '02', '04', '05', '13', '14'] # ignore the transition to neutral classes
					print(f"Using only classes relevant to the BLINC lab OI TMR control study:")
					print(f"\tNeutral, elbow flexion, elbow extension, pronation, supination, hand close, and hand open")
			else:
				if self.neutral_transitions:
					classes_values = [f"{i:02}" for i in range(15)] # classes 0 - 14
					print(f"Including neutral transition classes")
				else:
					classes_values = ['00', '01', '02', '04', '05', '07', '08', '10', '11', '13', '14'] # ignore the transition to neutral classes
					print(f"Ignoring neutral transition classes 3, 6, 9, and 12")
	
		print(f"Extracting full reachable workspace files with regex values:")
		print(f"\tsubjects_values = {subjects_values}")
		print(f"\tsets_values = {sets_values}")
		print(f"\treps_values = {reps_values}")
		print(f"\tclasses_values = {classes_values}")

		regex_filters = [
			RegexFilter(left_bound = "/", right_bound="/", values = sets_values, description='sets'), # trials 1-3 folders
			RegexFilter(left_bound = "ac", right_bound="_emg", values = classes_values, description='classes'), # only emg files for classes 
			RegexFilter(left_bound = "_lp", right_bound="_ac", values = reps_values, description='reps'), # limb positions is equal to reps for our dataset
			RegexFilter(left_bound="participant", right_bound="/",values=subjects_values, description='subjects') # participantXX folders
		]
		odh = OfflineDataHandler()
		# skip the first row as it contains a header
		# only grab the correct columns
		# the first column contains the time stamp so we need to skip that
		# braced is always first columns 1 - 7
		# unbraced is next columns 8 - 14
		if self.braced:
			print(f"Extracting data for BRACED arm")
			data_columns = [i for i in range(1, 8)] 
		else:
			print(f"Extracting data for UNBRACED arm")
			data_columns = [i for i in range(8, 15)]

		odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",", skiprows=1, data_column=data_columns, sort_files=True)
		# odh.get_data(folder_location=self.dataset_folder, regex_filters=regex_filters, delimiter=",", skiprows=1, data_column=data_columns, sort_files=False)
		
		if self.training_limb_positions == 1:
			# limb position variation (i.e., split by limb position not trial)
			train_reps = [0]
			test_reps = [1, 2, 3, 4, 5, 6, 7, 8]
			print(f"Splitting into train/test sets by limb position. \n\tTrain reps: {train_reps}\n\tTest reps: {test_reps}")
			odh_train = odh.isolate_data('reps', train_reps, fast=True)
			odh_test = odh.isolate_data('reps', test_reps, fast=True)
		
		elif self.training_limb_positions == 3:
			# limb position variation (i.e., split by limb position not trial)
			train_reps = [0, 3, 4]
			test_reps = [1, 2, 5, 6, 7, 8]
			# train_reps = [0, 1, 3, 4, 7]
			# test_reps = [2, 5, 6, 8]
			print(f"Splitting into train/test sets by limb position. \n\tTrain reps: {train_reps}\n\tTest reps: {test_reps}")
			odh_train = odh.isolate_data('reps', train_reps, fast=True)
			odh_test = odh.isolate_data('reps', test_reps, fast=True)
		
		elif self.training_limb_positions == 9:
			# use all limb positions in training data (i.e., split by trial)
			# train on trial 1; test on trials 2 & 3
			train_sets = [0]
			test_sets = [1, 2]
			print(f"Splitting into train/test sets by trial. \n\tTrain sets: {train_sets}\n\tTest sets: {test_sets}")
			odh_train = odh.isolate_data('sets', train_sets, fast=True)
			odh_test = odh.isolate_data('sets', test_sets, fast=True)
		
		elif self.train_participants:
			# split by participants
			if self.test_participants:
				# we have both train and test set designated
				train_subjects = [subjects_values.index(participant) for participant in self.train_participants]
				test_subjects = [subjects_values.index(participant) for participant in self.test_participants]
				print(f"Training on SPECIFIED participants: {self.train_participants} ({train_subjects})")
				print(f"Testing on SPECIFIED participants: {self.test_participants} ({test_subjects})")
				odh_train = odh.isolate_data('subjects', train_subjects, fast=True)
				odh_test = odh.isolate_data('subjects', test_subjects, fast=True)
			
			else:
				# test subjects not specified so lets use all the participants not in the training set
				train_subjects = [subjects_values.index(participant) for participant in self.train_participants]
				test_subjects = [subjects_values.index(participant) for participant in subjects_values if participant not in self.train_participants]

				print(f"Training on SPECIFIED participants: {self.train_participants} ({train_subjects})")
				print(f"Testing on REMAINING participants: {self.test_participants} ({test_subjects})")
				odh_train = odh.isolate_data('subjects', train_subjects, fast=True)
				odh_test = odh.isolate_data('subjects', test_subjects, fast=True)
		
		elif not self.train_participants:
			if self.test_participants:
				# train subjects not specified so lets use all the participants not in the test set
				train_subjects = [subjects_values.index(participant) for participant in subjects_values if participant not in self.test_participants]
				test_subjects = [subjects_values.index(participant) for participant in self.test_participants]

				print(f"Training on REMAINING participants: {self.train_participants} ({train_subjects})")
				print(f"Testing on SPECIFIED participants: {self.test_participants} ({test_subjects})")
				odh_train = odh.isolate_data('subjects', train_subjects, fast=True)
				odh_test = odh.isolate_data('subjects', test_subjects, fast=True)
			
			else:
				# neither specified, raise an warning
				print("WARNING! Test/training set split not specified. Setting split=False!")
				split = False
		
		else:
			# catch all other cases	
			print("ERROR this should never be reached! Figure out what's going on, Laura!")

		data = odh
		if split:
			data = {'All': odh, 'Train': odh_train, 'Test': odh_test}

		return data