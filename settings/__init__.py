import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS"))

SAMP_RATE = int(os.environ.get("SAMP_RATE", 20))
N_SMOOTH = int(os.environ.get("N_SMOOTH", 20))
Z_THRESHOLD = int(os.environ.get("Z_THRESHOLD", 6))
TURN_THRESHOLD = float(os.environ.get("TURN_THRESHOLD", 0.8))
LANE_CHANGE_THRESHOLD = float(os.environ.get("LANE_CHANGE_THRESHOLD", 0.6))

TN_THR = float(os.environ.get("TN_THR", 0.8))
LC_THR = float(os.environ.get("LC_THR", 0.6))
L1_THR = int(os.environ.get("L1_THR", 2))
L2_THR = int(os.environ.get("L2_THR", 3))
L3_THR = int(os.environ.get("L3_THR", 6))
L4_THR = int(os.environ.get("L4_THR", 12))
ACC_THR = int(os.environ.get("ACC_THR", 1))
ACC_FAC = int(os.environ.get("ACC_FAC", 3))
DEVICE_ID = os.environ.get("DEVICE_ID", 'iPad-001')
CALI_FILE = os.environ.get("CALI_FILE", 'calibration_matrix.csv')
EVT_DET_FILE = os.environ.get("EVT_DET_FILE", 'evt_detection_parameters.csv')
ACC_DET_FILE = os.environ.get("ACC_DET_FILE", 'acc_detection_parameters.csv')
RTT_EVA_FILE = os.environ.get("RTT_EVA_FILE", 'rtt_evaluation_parameters.csv')
LTT_EVA_FILE = os.environ.get("LTT_EVA_FILE", 'ltt_evaluation_parameters.csv')
UTN_EVA_FILE = os.environ.get("UTN_EVA_FILE", 'utn_evaluation_parameters.csv')
LCR_EVA_FILE = os.environ.get("LCR_EVA_FILE", 'lcr_evaluation_parameters.csv')
LCL_EVA_FILE = os.environ.get("LCL_EVA_FILE", 'lcl_evaluation_parameters.csv')
ACC_EVA_FILE = os.environ.get("ACC_EVA_FILE", 'acc_evaluation_parameters.csv')
CODE_FILE = os.environ.get("CODE_FILE", 'code_sys.xlsx')