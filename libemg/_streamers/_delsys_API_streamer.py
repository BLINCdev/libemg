"""
This is the class that handles the data that is output from the Delsys Trigno Base.
Create an instance of this and pass it a reference to the Trigno base for initialization.
See CollectDataController.py for a usage example.
"""
import numpy as np
from libemg.shared_memory_manager import SharedMemoryManager
from multiprocessing import Process, Event, Lock
import traceback

class DataKernel():
    def __init__(self, trigno_base):
        self.TrigBase = trigno_base
        self.packetCount = 0
        self.sampleCount = 0
        self.allcollectiondata = [[]]
        self.channel1time = []

    def processData(self, data_queue):
        """Processes the data from the DelsysAPI and place it in the data_queue argument"""
        outArr = self.GetData()
        if outArr is not None:
            for i in range(len(outArr)):
                self.allcollectiondata[i].extend(outArr[i][0].tolist())
            try:
                for i in range(len(outArr[0])):
                    if np.asarray(outArr[0]).ndim == 1:
                        data_queue.append(list(np.asarray(outArr, dtype='object')[0]))
                    else:
                        data_queue.append(list(np.asarray(outArr, dtype='object')[:, i]))
                try:
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
                except:
                    pass
            except IndexError:
                pass

    def processYTData(self, data_queue):
        """Processes the data from the DelsysAPI and place it in the data_queue argument"""
        outArr = self.GetYTData()
        if outArr is not None:
            for i in range(len(outArr)):
                self.allcollectiondata[i].extend(outArr[i][0].tolist())
            try:
                yt_outArr = []
                for i in range(len(outArr)):
                    chan_yt = outArr[i]
                    chan_ydata = np.asarray([k.Item2 for k in chan_yt[0]], dtype='object')
                    yt_outArr.append(chan_ydata)

                data_queue.append(list(yt_outArr))

                try:
                    self.packetCount += len(outArr[0])
                    self.sampleCount += len(outArr[0][0])
                except:
                    pass
            except IndexError:
                pass

    def GetData(self):
        """ Check if data ready from DelsysAPI via Aero CheckDataQueue() - Return True if data is ready
            Get data (PollData)
            Organize output channels by their GUID keys

            Return array of all channel data
        """

        dataReady = self.TrigBase.CheckDataQueue()                      # Check if DelsysAPI real-time data queue is ready to retrieve
        if dataReady:
            DataOut = self.TrigBase.PollData()                          # Dictionary<Guid, List<double>> (key = Guid (Unique channel ID), value = List(Y) (Y = sample value)
            outArr = [[] for i in range(len(DataOut.Keys))]             # Set output array size to the amount of channels being outputted from the DelsysAPI

            channel_guid_keys = list(DataOut.Keys)                      # Generate a list of all channel GUIDs in the dictionary
            for j in range(len(DataOut.Keys)):                          # loop all channels
                chan_data = DataOut[channel_guid_keys[j]]               # Index a single channels data from the dictionary based on unique channel GUID (key)
                outArr[j].append(np.asarray(chan_data, dtype='object')) # Create a NumPy array of the channel data and add to the output array

            return outArr
        else:
            return None

    def GetYTData(self):
        """ YT Data stream only available when passing 'True' to Aero Start() command i.e. TrigBase.Start(True)
            Check if data ready from DelsysAPI via Aero CheckYTDataQueue() - Return True if data is ready
            Get data (PollYTData)
            Organize output channels by their GUID keys

            Return array of all channel data
        """

        dataReady = self.TrigBase.CheckYTDataQueue()                        # Check if DelsysAPI real-time data queue is ready to retrieve
        if dataReady:
            DataOut = self.TrigBase.PollYTData()                            # Dictionary<Guid, List<(double, double)>> (key = Guid (Unique channel ID), value = List<(T, Y)> (T = time stamp in seconds Y = sample value)
            outArr = [[] for i in range(len(DataOut.Keys))]                 # Set output array size to the amount of channels being outputted from the DelsysAPI

            channel_guid_keys = list(DataOut.Keys)                          # Generate a list of all channel GUIDs in the dictionary
            for j in range(len(DataOut.Keys)):                              # loop all channels
                chan_yt_data = DataOut[channel_guid_keys[j]]                # Index a single channels data from the dictionary based on unique channel GUID (key)
                outArr[j].append(np.asarray(chan_yt_data, dtype='object'))  # Create a NumPy array of the channel data and add to the output array

            return outArr
        else:
            return None
        

class DelsysAPIStreamer(Process):
    def __init__(self, key, license, dll_folder = 'resources/', shared_memory_items: list = [],
                       emg: bool = True,
                       imp: bool = False,
                       imu: bool = False):
        Process.__init__(self, daemon=True)

        self.dll_folder = dll_folder
        self.connected = False
        self.signal = Event()
        self.recording_signal = Event()
        self.shared_memory_items = shared_memory_items
        self.key = key 
        self.license = license
        
        self.emg = emg
        self.imp = imp
        self.imu = imu
        
        self.emg_handlers = []
        self.imp_handlers = []
        self.imu_handlers = []
    
    def connect(self, key, license):
        try:
            self.trigbase.ValidateBase(key, license)
        except Exception as e:
            if "product not licensed." in str(e):
                print("Error: Key/License Not Validated\nClose the program and paste your key/license into TrignoBase.py file\nContact support@delsys.com if you have not received your APi key/license")
            elif "no RF subsystem found" in str(e):
                print("Error: Trigno system not found\nPlease make sure your base station or lite dongle is plugged in via USB\nVisit our website to request a quote or contact support@delsys.com")
            else:
                print(str(e))
            print(Exception)
        
        self.sensors = self.scan()
        self.select_sensors()

    def scan(self):
        '''
        outArr = outArr[12, 2, 4, 6, 13, 7, 10, 5, 9, 14, 11, 8, 3, 1]
        1 slots (starting at 12) occupied by sensor 85894 with sticker number 12
        1 slots (starting at 2) occupied by sensor 85619 with sticker number 2
        1 slots (starting at 4) occupied by sensor 85890 with sticker number 4
        1 slots (starting at 6) occupied by sensor 85768 with sticker number 6
        1 slots (starting at 13) occupied by sensor 85829 with sticker number 13
        1 slots (starting at 7) occupied by sensor 85645 with sticker number 7
        1 slots (starting at 10) occupied by sensor 85854 with sticker number 10
        1 slots (starting at 5) occupied by sensor 85639 with sticker number 5
        1 slots (starting at 9) occupied by sensor 85853 with sticker number 9
        1 slots (starting at 14) occupied by sensor 85784 with sticker number 14
        1 slots (starting at 11) occupied by sensor 85840 with sticker number 11
        1 slots (starting at 8) occupied by sensor 85653 with sticker number 8
        1 slots (starting at 3) occupied by sensor 85949 with sticker number 3
        1 slots (starting at 1) occupied by sensor 85884 with sticker number 1
        '''
        try:
            f = self.trigbase.ScanSensors().Result
            print(f"ScanSensors: {f}")
            # for i in range(14):
            #     self.trigbase.SelectSensor(i)
            #     f = self.trigbase.GetScannedSensorsFound()
            #     print(f"SENSORS SENSORS SENSORS Found: {f}")
        except Exception as e:
            print(str(e))

        all_scanned_sensors = self.trigbase.GetScannedSensorsFound()
        print("Sensors Found:\n")
        print(f"all_scanned_sensors: {all_scanned_sensors}")
        # for sensor in all_scanned_sensors:
        self.sensor_order = []
        for i in range(len(all_scanned_sensors)):
            sensor = self.trigbase.GetSensorObject(i)
            print(str(i) + "(" + str(sensor.PairNumber) + ") " +
                sensor.FriendlyName + "\n" +
                sensor.Configuration.ModeString + "\n")
            self.sensor_order.append(sensor.PairNumber-1)
        return all_scanned_sensors

    def select_sensors(self):
        sensor_count = len(self.sensors)
        for i in range(sensor_count):
            self.trigbase.SelectSensor(i)

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)

    def add_imp_handler(self, h):
        self.imp_handlers.append(h)

    def add_imu_handler(self, h):
        self.imu_handlers.append(h)    

    def start_recording(self):
        """Start writing data to the shared memory buffer."""
        print("Started writing to buffer")
        self.recording_signal.set()

    def stop_recording(self):
        """Stop writing data to the shared memory buffer."""
        print("Stopped writing to buffer")
        self.recording_signal.clear()

    def run(self):
        try:
            from collections import deque
            from pythonnet import load
            load("coreclr")
            import clr
        except RuntimeError as e:
            raise RuntimeError('.NET runtime not found, so DelsysAPI Streamer cannot run. Please ensure that a .NET runtime >8.0 is installed. Exiting run() method.') from e

        clr.AddReference(self.dll_folder + "DelsysAPI")
        clr.AddReference("System.Collections")
        from Aero import AeroPy
        # Set up shared memory 
        self.trigbase = AeroPy()
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        # Start recording to shared memory manager as default behaviour
        self.start_recording()

        # Set up handlers for shared memory
        def write_emg(emg):
            if self.recording_signal.is_set():
                # update the samples in "emg"
                self.smm.modify_variable("emg", lambda x: np.vstack((np.flip(emg,0), x))[:x.shape[0],:])
                # update the number of samples retrieved
                self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])
        self.add_emg_handler(write_emg)

        def write_imp(imp):
            if self.recording_signal.is_set():
                self.smm.modify_variable("imp", lambda x: np.vstack((np.flip(imp,0), x))[:x.shape[0],:])
                self.smm.modify_variable("imp_count", lambda x: x + imp.shape[0])
        self.add_imp_handler(write_imp)

        def write_imu(imu):
            if self.recording_signal.is_set():
                self.smm.modify_variable("imu", lambda x: np.vstack((np.flip(imu,0), x))[:x.shape[0],:])
                self.smm.modify_variable("imu_count", lambda x: x + imu.shape[0])
        self.add_imu_handler(write_imu)

        self.connect(self.key, self.license)

        if self.trigbase.GetPipelineState() == 'Connected':
            self.trigbase.Configure(False, False)
            channelcount = 0
            channelobjects = []
            datahandler = DataKernel(self.trigbase)
            emg_idxs = []
            imp_idxs = []
            imu_idxs = []

            for i in range(len(self.sensors)):
                print(f"INIT Checking modes for sensor {i}: {self.sensors[i].FriendlyName}")
                # sample_modes = self.getSampleModes(i)
                # print(f"\tSample modes for sensor {i}: {sample_modes}") # careful this is a long list lol
                print(f"\tCurrent mode for sensor {i}: {self.getCurMode(i)}")

                selectedSensor = self.trigbase.GetSensorObject(i)
                print("(" + str(selectedSensor.PairNumber) + ") " + str(selectedSensor.FriendlyName))

                if len(selectedSensor.TrignoChannels) > 0:
                    print("--Channels")

                    for channel in range(len(selectedSensor.TrignoChannels)):
                        sample_rate = round(selectedSensor.TrignoChannels[channel].SampleRate, 3)
                        print("----" + selectedSensor.TrignoChannels[channel].Name + " (" + str(sample_rate) + " Hz)")
                        channelcount += 1
                        channelobjects.append(channel)
                        datahandler.allcollectiondata.append([])

                        emg_idxs.append("EMG" in selectedSensor.TrignoChannels[channel].Name)
                        imp_idxs.append("IMP" in selectedSensor.TrignoChannels[channel].Name)
                        imu_idxs.append(any(x in selectedSensor.TrignoChannels[channel].Name for x in ["ACC", "GYRO"]))
            # print(f"EMG idxs: {emg_idxs}")
            # print(f"IMP idxs: {imp_idxs}")
            # print(f"IMU idxs: {imu_idxs}")
            self.trigbase.Start(False)
            # sensor_order = [11, 1, 3, 5, 12, 6, 9, 4, 8, 13, 10, 7, 2, 0]
            while True:
                try:
                    """
                    EXPECTED SHAPES
                    EMG data shape: (34, 14)
                    IMP data shape: (2, 14)
                    imu_data shape: (4, 84)
                    """
                    outArr = datahandler.GetData()
                    
                    # print(f"outArr: {outArr}")  
                    # convert to one single np array
                    if outArr is not None:
                        # print(f"shape of outArr: {len(outArr)}")
                        # print(f"outArr: {outArr}")
                        # # reorder outArr to match the order of the paired sensors by number
                        # outArr = [outArr[i] for i in [11, 1, 3, 5, 12, 6, 9, 4, 8, 13, 10, 7, 2, 0]] 
                        # print(f"shape of NEW outArr: {len(outArr)}")
                        # print(f"NEW outArr: {outArr}")
                        # exit()
                        # ------------------------
                        # OLD WORKING CODE
                        # emg_data = []
                        # imp_data = []
                        # imu_data = []
                        # # First pass: collect all IMU data and find the minimum length
                        # imu_channel_data = []
                        # min_length = float('inf')
                        # for i in range(len(outArr)):
                        #     if emg_idxs[i]:
                        #         emg_data.append(outArr[i][0])
                        #     elif imp_idxs[i]:
                        #         imp_data.append(outArr[i][0])
                        #     elif imu_idxs[i]:
                        #         channel_data = np.array(outArr[i][0])
                        #         if channel_data.ndim == 1:
                        #             channel_data = channel_data.reshape(-1, 1)
                        #         imu_channel_data.append(channel_data)
                        #         min_length = min(min_length, len(channel_data))
                        #         # print(f"IMU Channel {i} data shape: {channel_data.shape}")
                        #     else:
                        #         print(f"ERROR: Channel {i} is not EMG, IMP, or IMU")
                        # if imu_channel_data:
                        #     imu_data = np.hstack([data[:min_length] for data in imu_channel_data])
                        #     print(f"Combined IMU data shape: {imu_data.shape}")
                        # ------------------------------
                        
                        emg_data = [0]*len(self.sensor_order)
                        imp_data = [0]*len(self.sensor_order)
                        imu_data = []
                        # imu_channel_data = []
                        # First pass: collect all IMU data and find the minimum length
                        imu_channel_data = [0]*len(self.sensor_order)*6
                        min_length = float('inf')
                        imu_counter = 0
                        for i in range(len(outArr)):
                            if emg_idxs[i]:
                                emg_data[self.sensor_order[i//8]] = outArr[i][0]
                            elif imp_idxs[i]:
                                imp_data[self.sensor_order[i//8]] = outArr[i][0]
                            elif imu_idxs[i]:
                                channel_data = np.array(outArr[i][0])
                                if channel_data.ndim == 1:
                                    channel_data = channel_data.reshape(-1, 1)
                                # imu_channel_data.append(channel_data)
                                imu_channel_data[self.sensor_order[i//8] * 6 + imu_counter] = channel_data
                                imu_counter += 1
                                imu_counter = imu_counter % 6
                                min_length = min(min_length, len(channel_data))
                                # print(f"IMU Channel {i} data shape: {channel_data.shape}")
                            else:
                                print(f"ERROR: Channel {i} is not EMG, IMP, or IMU")
                        
                        # Second pass: trim all channels to the minimum length
                        if imu_channel_data:
                            imu_data = np.hstack([data[:min_length] for data in imu_channel_data])
                            # print(f"Combined IMU data shape: {imu_data.shape}")
                        
                        # handle emg data
                        emg_data = np.array(emg_data).T
                        if emg_data.shape[1] == 1:
                            emg_data = emg_data[:,None]
                        # print(f"EMG data shape: {emg_data.shape}")
                        
                        for e in self.emg_handlers:
                            e(emg_data)

                        # handle imp data
                        imp_data = np.array(imp_data).T
                        if imp_data.shape[1] == 1:
                            imp_data = imp_data[:,None]
                        # print(f"IMP data shape: {imp_data.shape}")
                        for i in self.imp_handlers:
                            i(imp_data)

                        # handle imu data
                        # imu_data = np.array(imu_data).T
                        # print(f"imu_data shape: {imu_data.shape}")
                        if imu_data.size > 0:  # Only process if we have IMU data
                            try:
                                for i in self.imu_handlers:
                                    i(imu_data)
                            except Exception as e:
                                print(f"Error processing IMU data: {str(e)}")
                                print(f"Error type: {type(e)}")
                                print(f"IMU data shape: {imu_data.shape}")
                                print(f"IMU data type: {imu_data.dtype}")
                                print(f"IMU data sample: {imu_data[:5]}")  # Print first 5 rows
                                print(f"Full traceback: {traceback.format_exc()}")
                except Exception as e:
                    print("LibEMG -> DelsysAPIStreamer: Error ocurred " + str(e))
                    print(f"Error type: {type(e)}")
                    print(f"Full traceback: {traceback.format_exc()}")
                    print(f"outArr type: {type(outArr)}")
                    if outArr is not None:
                        print(f"outArr length: {len(outArr)}")
                        for i, arr in enumerate(outArr):
                            print(f"Channel {i} type: {type(arr)}")
                            if len(arr) > 0:
                                print(f"Channel {i} data type: {type(arr[0])}")
                if self.signal.is_set():
                    self.cleanup()
                    break
            print("LibEMG -> DelsysStreamer (process ended).")

    # ---------------------------------------------------------------------------------
    # ---- Helper Functions

    def getSampleModes(self, sensorIdx):
        """Gets the list of sample modes available for selected sensor"""
        sampleModes = self.trigbase.AvailibleSensorModes(sensorIdx)
        # Convert .NET array to Python list
        return [str(mode) for mode in sampleModes]

    def getCurMode(self, sensorIdx):
        """Gets the current mode of the sensors"""
        if sensorIdx >= 0 and sensorIdx < len(self.sensors):
            curModes = self.trigbase.GetCurrentSensorMode(sensorIdx)
            return curModes
        else:
            return None

    def setSampleMode(self, curSensor, setMode):
        """Sets the sample mode for the selected sensor"""
        self.trigbase.SetSampleMode(curSensor, setMode)
        mode = self.getCurMode(curSensor)
        sensor = self.trigbase.GetSensorObject(curSensor)
        if mode == setMode:
            print("(" + str(sensor.PairNumber) + ") " + str(sensor.FriendlyName) +" Mode Change Successful")

    def cleanup(self):
        pass

    def __del__(self):
        pass
