import sys
import os

def generate_cafconfig(path, processes_num, launch_file, machine_file, config_name='cafconfig.txt'):
    with open(os.path.join(path, config_name), 'w') as f:
        f.write('-genvall -genv I_MPI_FABRICS=shm:ofi -machinefile {0} -n {1} {2}'.format(
            machine_file, processes_num, os.path.join(path, launch_file)))

if __name__ == '__main__':
    if len(sys.argv) == 3:
        node = sys.argv[1]
        path = sys.argv[2]

        generate_cafconfig(path, 4, 'iPM-IFE', node)
        os.system(os.path.join(path, 'iPM-IFE'))

    else:
        print('The number of input parameters is incorrect.')
        sys.exit(-1)
