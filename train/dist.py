import os

def is_master():
  rank = int(os.environ.get("RANK", 0))
  return rank == 0