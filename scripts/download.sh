# download data to data/
mkdir -p data
gdown 1qgfMiKG2jKfgrjLuCjuk3MNoSaYF1_Ro -O data/processed_data.tar.gz
tar -xf data/processed_data.tar.gz -C data/

# download model to output/
mkdir -p output
gdown 1J-p4qlpzGuRWxDHoDjGkHcG4gtDgu-qX -O output/model.tar.gz
tar -xf output/model.tar.gz -C output/

# copy 
mkdir -p data/lib
cp docs/*.json data/lib/