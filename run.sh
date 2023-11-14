for i in {8..30..2}
do
    echo $i
    make clean
    make all EXPERIMENT="-DnParticles_pow=$i "
    build/base >> data/CPU.dat
    build/base_cu >> data/GPU_kernel_1.dat
    build/base_kernel2 >> data/GPU_kernel_2.dat
done
