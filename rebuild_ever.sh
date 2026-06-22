# make sure this is run from the correct directory by checking for ever subdir
if [ ! -d "ever" ]; then 
    echo "No ever dir!"
    exit
else
    echo "Building ever"   
fi
source activate ever
rm -rf ever/build
mkdir ever/build 
cd ever/build
cmake -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86" ..
make -j8
