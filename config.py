import time
MIN_PRUNED_SCORE = 0.65
N_PARENTS = 3
N_SIBLINGS = 1
N_VAR_SIBLINGS = 2
NUM_SIMILARS = 100
MIN_SIMILARITY_SCORE = 0.4
VOCAB_FILE = "vocab.pkl"
TFIDF_FILE = "tfidf.pkl"
NUM_FEATURE_MIN = 10
SAMPLE_METHOD_MIN_LINES = 15
SEED = time.time()
N_SAMPLES = 100
IGNORE_VAR_NAMES = True
IGNORE_SIBLING_FEATURES = False
IGNORE_VAR_SIBLING_FEATURES = False
CLUSTER = False
PRINT_SIMILAR = True
THRESHOLD1 = 0.9
THRESHOLD2 = 1.5
TOP_N = 5
TEST_ALL = False
PRINT_TEST = True
RECORD_QUANTITY = 0
g_vocab = None