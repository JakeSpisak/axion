import pstats
import sys

input_file = sys.argv[1]
p = pstats.Stats(input_file)
p.sort_stats('cumtime').print_stats(30)