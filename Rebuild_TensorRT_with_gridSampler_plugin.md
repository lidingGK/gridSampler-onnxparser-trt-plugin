# Rebuild TensorRT with gridSampler plugin

`Based on image nvcr.io/nvidia/pytorch:20.08-py3`

```bash
cd ~
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT
# Checkout to proper version
git checkout origin/release/7.1
git submodule update --init --recursive

cd ~
git clone https://github.com/lidingGK/onnxparser-trt-plugin-sample.git
cp -ru  onnxparser-trt-plugin-sample/TensorRT/* TensorRT/

export TRT_OSSPATH=~/TensorRT
export TRT_LIBPATH=/usr/lib/x86_64-linux-gnu

cd $TRT_OSSPATH
mkdir -p build && cd build
# Notify the platform and cuda version
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=x86_64 -DCUDA_VERSION=11.0
make -j$(nproc)

cp out/libnvcaffeparser.so.7.1.3 $TRT_LIBPATH
cp out/libnvinfer_plugin.so.7.1.3 $TRT_LIBPATH
cp out/libnvonnxparser.so*  $TRT_LIBPATH

cd $TRT_LIBPATH
rm libnvonnxparser.so.7
ln -s libnvonnxparser.so.7.0.0 libnvonnxparser.so.7

cd ~
pip install pycuda
pip install nvidia-pyindex
pip install onnx-graphsurgeon

# Test Plugin
LD_PRELOAD=$TRT_OSSPATH/build/out/libnvinfer_plugin.so python ~/onnxparser-trt-plugin-sample/test_plugin_result.py
python ~/onnxparser-trt-plugin-sample/test_plugin_result.py

```

Run AnyNet

```bash
git clone git@github.com:lidingGK/AnyNet.git
cd AnyNet/model/spn_t1
sh make.sh
cd ../..
cp -r /workspace/AnyNet.bk/data ./
cp -r /workspace/AnyNet.bk/checkpoint ./
python finetune.py --maxdisp 192 --with_spn --datapath data/training/ \
    --save_path results/kitti2015 --datatype 2015 --pretrained checkpoint/kitti2015_ck/checkpoint.tar \
    --split_file checkpoint/kitti2015_ck/split.txt --evaluate


```

