mkdir -p data/raw

# download data
git clone https://github.com/CLC-HCMUS/ViMs-Dataset.git data/raw
unzip data/raw/ViMs.zip -d data/raw
rm data/raw/ViMs.zip
rm data/raw/README.md
rm -rf data/raw/__MACOSX
rm -rf data/raw/.git

# extract content
mkdir -p data/interim
python src/data/preprocess/get_content.py