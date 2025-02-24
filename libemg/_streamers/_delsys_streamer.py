import socket
import selectors
import struct
import numpy as np
from libemg.shared_memory_manager import SharedMemoryManager
from multiprocessing import Process, Event

class DelsysEMGStreamer(Process):
    """
    Delsys Trigno wireless EMG system.

    Requires the Trigno Control Utility to be running.

    ----------
    shared_memory_items : list
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype, Lock()].
    emg : bool
        Enable EMG streaming
    imu : bool
        Enable IMU streaming
    recv_ip : str
        The ip address the device is connected to (likely 'localhost').
    cmd_port : int
        Port of TCU command messages.
    data_port : int
        Port of TCU data access.
    total_channels : int
        Total number of channels supported by the device.
    timeout : float
        Number of seconds before socket returns a timeout exception

    Attributes
    ----------
    BYTES_PER_CHANNEL : int
        Number of bytes per sample per channel. EMG and accelerometer data
    CMD_TERM : str
        Command string termination.

    Notes
    -----
    Implementation details can be found in the Delsys SDK reference:
    http://www.delsys.com/integration/sdk/
    """

    BYTES_PER_CHANNEL = 4
    CMD_TERM = '\r\n\r\n'
    EMG_PORT = 50043
    IMU_PORT = 50044

    def __init__(self, 
                 shared_memory_items:   list = [],
                 emg=True,
                 imu=False,
                 recv_ip:               str = 'localhost', 
                 cmd_port:              int = 50040,
                 data_port:             int = 50043,
                 aux_port:              int = 50044,
                 channel_list:          list[int] = list(range(8)),
                 timeout:               int = 10):
        """
        Note: data_port 50043 refers to the current port that EMG data is being streamed to. For older devices, the EMG data_port may be 50041 (e.g., the Delsys Trigno)
        """
        Process.__init__(self)

        self.connected = False
        self.signal = Event()
        self.recording_signal = Event()
        self.shared_memory_items = shared_memory_items

        self.emg = emg
        self.imu = imu

        self.emg_handlers = []
        self.imu_handlers = []

        self.host = recv_ip
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.aux_port = aux_port
        self.channel_list = channel_list
        self.aux_channel_list = list(range((channel_list[-1]+1)*9))
        self.timeout = timeout

        self._min_recv_size = 16 * self.BYTES_PER_CHANNEL
        self._min_recv_size_aux = 144 * self.BYTES_PER_CHANNEL

        # Initialize these in run()
        self._comm_socket = None
        self._data_socket = None
        self._aux_socket = None
        self.smm = None

    def connect(self):
        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)
        self._data_socket.setblocking(False)

        # create the aux data socket
        self._aux_socket = socket.create_connection(
            (self.host, self.aux_port), self.timeout)
        self._aux_socket.setblocking(False)

        self._send_cmd('START')
        self.connected = True

    def add_emg_handler(self, h):
        self.emg_handlers.append(h)
    
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

    def _emg_stream(self):
        """Thread function to handle EMG data"""
        try:
            packet = self._data_socket.recv(self._min_recv_size)
            data = np.asarray(struct.unpack('<'+'f'*16, packet))
            data = data[self.channel_list]
            if len(data.shape)==1:
                data = data[None, :]
            for e in self.emg_handlers:
                e(data)
        except Exception as e:
            print(f"EMG Stream Error: {str(e)}")
            if "timed out" in str(e): # Check if the exception is a timeout
                self.connected = False
                return


    def _imu_stream(self):
        """Thread function to handle IMU data"""
        try:
            packet = self._aux_socket.recv(self._min_recv_size_aux)
            data = np.asarray(struct.unpack('<'+'f'*144, packet))
            data = data[self.aux_channel_list]
            if len(data.shape)==1:
                data = data[None, :]
            for i in self.imu_handlers:
                i(data)
        except Exception as e:
            print(f"IMU Stream Error: {str(e)}")
            if "timed out" in str(e): # Check if the exception is a timeout
                self.connected = False
                return

    def run(self):
        # Initialize shared memory manager
        self.smm = SharedMemoryManager()
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        # Set up handlers for shared memory
        def write_emg(emg):
            if self.recording_signal.is_set():
                self.smm.modify_variable("emg", lambda x: np.vstack((np.flip(emg,0), x))[:x.shape[0],:])
                self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])
        self.add_emg_handler(write_emg)

        def write_imu(imu):
            if self.recording_signal.is_set():
                self.smm.modify_variable("imu", lambda x: np.vstack((np.flip(imu,0), x))[:x.shape[0],:])
                self.smm.modify_variable("imu_count", lambda x: x + imu.shape[0])
        self.add_imu_handler(write_imu)

        # Connect to device
        try:
            self.connect()
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return

        sel = selectors.DefaultSelector()
        emg_key: str = "EMG"
        aux_key: str = "AUX"
        sel.register(self._data_socket, selectors.EVENT_READ, data=emg_key)
        sel.register(self._aux_socket, selectors.EVENT_READ, data=aux_key)

        # Main process loop
        while not self.signal.is_set():
            events = sel.select()
            for key, mask in events:
                # The main expected terminator of the reading loop is the lock-backed signal, but internally, the
                # stream reading methods try-catch stream errors and ask for the loop to be terminated to handle, e.g.,
                # unexpected server termination. In these cases, we want to be able to bail out early.
                if not self.connected:
                    break

                event_key: str = key.data

                if event_key == emg_key:
                    self._emg_stream()
                elif event_key == aux_key:
                    self._imu_stream()
                else:
                    # Should be unreachable.
                    raise ValueError(f"Unknown selector key: {event_key}")

        # Cleanup
        print("Reached cleanup")
        self.connected = False

        self.cleanup()
        print("LibEMG -> DelsysStreamer (process ended).")

    def cleanup(self):
        if self._comm_socket:
            try:
                self._send_cmd('STOP')
                print("LibEMG -> DelsysStreamer (streaming stopped).")
                self._comm_socket.close()
                print("LibEMG -> DelsysStreamer (comm socket closed).")
                self._comm_socket = None
            except Exception as e:
                print(f"Error sending STOP command: {e}")
        
        if self._data_socket:
            try:
                self._data_socket.close()
                self._data_socket = None
            except:
                pass
            
        if self._aux_socket:
            try:
                self._aux_socket.close()
                self._aux_socket = None
            except:
                pass

    def __del__(self):
        self.cleanup()

    def _send_cmd(self, command):
        self._comm_socket.send(self._cmd(command))
        resp = self._comm_socket.recv(128)
        self._validate(resp)

    @staticmethod
    def _cmd(command):
        return bytes("{}{}".format(command, DelsysEMGStreamer.CMD_TERM),
                     encoding='ascii')

    @staticmethod
    def _validate(response):
        s = str(response)
        if 'OK' not in s:
            print("warning: TrignoDaq command failed: {}".format(s))

    