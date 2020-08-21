cd ../code
python dct.py -lp ../data/lmdbFiles/BigEarthNet-19.lmdb -tr ../data/train.csv -vl ../data/val.csv -te ../data/test.csv -b 32 -e 6 -nt symmetry -nr 0.5 -ch RGB -lb BigEarthNet-19 -si 100000.0 -sr 0.25 -lto 1.0 -ltr 1.0
