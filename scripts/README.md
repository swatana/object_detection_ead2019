# making graphs

## bboxと拡大したbboxの輝度値の平均と分散を求める
```
python3 scripts/image_filter.py -g=data_labels/bbox/ead2019_004/ground-truth -f=sobel -r=0.1
```
## クラスごとのbboxのサイズのヒストグラムを作成
```
python3 size_histogram.py -g=data_labels/bbox/ead2019_004/ground-truth t=data_labels/bbox/ead2019_004/test.txt
```
# wndchrm 
## wndchrm install 
```
sudo apt -y install build-essential gcc g++ make libtool texinfo dpkg-dev pkg-config
wget http://www.fftw.org/fftw-3.3.10.tar.gz
tar -xvzof fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --prefix=/usr/local/ --enable-threads
make
sudo make install
cd ..

wget http://download.osgeo.org/libtiff/tiff-4.5.0rc3.tar.gz
tar -xvzof tiff-4.5.0rc3.tar.gz
cd tiff-4.5.0
./configure
make
sudo make install
cd ..

git clone https://gitlab.com/iggman/wnd-charm.git
cd wnd-charm/
./configure
make
sudo make install
```
## wndchrm用にgtのbboxのtif画像を生成する
```
python3 scripts/gt_crop_tif.py -g=data_labels/bbox/ead2019_004/ground-truth -t=data_labels/bbox/ead2019_004/test.txt
```
## wndchrm 実行
```
wndchrm train -m -l crop crop.fit
```
## wndchrmの結果から指定した特徴量ごとの平均を作成
```
python3 scripts/cal_average_feature.py -f=Gini Coefficient () -r=result
```
## wndchrmの結果から特徴量ごとのヒストグラムを求める
python3 scripts/hist_feature.py
