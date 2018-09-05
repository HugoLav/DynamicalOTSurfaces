#!/bin/sh
echo "Hand mesh (alpha: 0.02, ntime: 13)"
python3 run_example.py --mesh=meshes/hand_3k.off --boundary=meshes/hand.bdy --alpha=0.02 --ntime=13
echo "*************"
echo "Hand mesh (alpha: 0.02, ntime: 31)"
python3 run_example.py --mesh=meshes/hand_3k.off --boundary=meshes/hand.bdy --alpha=0.02 --ntime=31
echo "-------------"

echo "Punctured sphere mesh (alpha: 0.02, ntime: 13)"
python3 run_example.py --mesh=meshes/sphere_puncture.off --boundary=meshes/sphere_puncture.bdy --alpha=0.02 --ntime=13
echo "*************"
echo "Punctured sphere mesh (alpha: 0.02, ntime: 31)"
python3 run_example.py --mesh=meshes/sphere_puncture.off --boundary=meshes/sphere_puncture.bdy --alpha=0.02 --ntime=31
echo "-------------"

echo "Armadillo mesh (alpha: 0.0, ntime: 31)"
python3 run_example.py --mesh=meshes/armadillo.off --boundary=meshes/armadillo.bdy --alpha=0.0 --ntime=31
echo "*************"
echo "Armadillo mesh (alpha: 0.0, ntime: 63)"
python3 run_example.py --mesh=meshes/armadillo.off --boundary=meshes/armadillo.bdy --alpha=0.0 --ntime=63
echo "*************"
echo "Armadillo mesh (alpha: 1, ntime: 31)"
python3 run_example.py --mesh=meshes/armadillo.off --boundary=meshes/armadillo.bdy --alpha=1 --ntime=31
echo "-------------"

echo "Airplane mesh (alpha: 0.1, ntime: 31)"
python3 run_example.py --mesh=meshes/airplane_62.off --boundary=meshes/airplane_62.bdy --alpha=0.1 --ntime=31
echo "-------------"

echo "Face mesh (alpha: 0.1, ntime: 31)"
python3 run_example.py --mesh=meshes/face_vector_field_319.off --boundary=meshes/face_vector_field_319.bdy --alpha=0.1 --ntime=31
echo "-------------"

echo "Square mesh (alpha: 0.0, ntime: 31)"
python3 run_example.py --mesh=meshes/square_regular_100.off --boundary=meshes/square_regular_100.bdy --alpha=0.0 --ntime=31
