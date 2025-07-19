echo "Creating target directory.."
mkdir -p /mnt/disk6/daiki/Datasets/iKAT/ikat_demo
mkdir -p /mnt/disk6/daiki/Datasets/iKAT/ikat_demo/collection
echo "Created target directory."

# JDKのインストール
conda install -c conda-forge openjdk=21 -y
# condaのbase環境を有効化してJDKを利用可能にする
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base