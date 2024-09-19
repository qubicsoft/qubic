import signal, traceback, os, string
from progressbar import ProgressBar, Bar, ETA, Percentage
import numpy as np
from PIL import Image
from qubic.data import PATH as data_dir
from qubic.calfiles import PATH as cal_dir
from qubic.dicts import PATH as dicts_dir


_NLEVELS = 0
_MAX_NLEVELS = 0

def join_toward_rank(comm, data, target_rank):
    #print('enter', target_rank)
    gathered_data = comm.gather(data, root=target_rank)
    #print('bis')
    if comm.Get_rank() == target_rank:
        #print(' bis bis')
        return np.concatenate(gathered_data)#[0]
    else:
        return
    


def join_data(comm, data):

    if comm is None:
        pass
    else:
        data = comm.gather(data, root=0)

        if comm.Get_rank() == 0:

            data = np.concatenate(data)
        
        data = comm.bcast(data)

    return data

def split_data(comm, theta):
    if comm is None:
        return theta
    else:
        return np.array_split(theta, comm.Get_size())[comm.Get_rank()]


class _ProgressBar(ProgressBar):
    def __init__(self, poll=0.1, **keywords):
        global _NLEVELS, _MAX_NLEVELS
        self._nlevels = _NLEVELS
        _NLEVELS += 1
        _MAX_NLEVELS = max(_MAX_NLEVELS, _NLEVELS)
        ProgressBar.__init__(self, poll=poll, **keywords)
        if self._nlevels == 0:
            self._signal_old = signal.signal(signal.SIGINT, self._int_handler)
        print('\n\n\033[F', end='')

    def update(self, n=None):
        if n is not None:
            ProgressBar.update(self, n)
            return
        ProgressBar.update(self, self.currval + 1)
        if self.currval >= self.maxval:
            self.finish()

    def finish(self):
        global _NLEVELS, _MAX_NLEVELS
        ProgressBar.finish(self)
        if self._nlevels == 0:
            print((_MAX_NLEVELS - 1) * '\n', end='')
            signal.signal(signal.SIGINT, self._signal_old)
            _MAX_NLEVELS = 0
        else:
            print('\033[F\033[F', end='')
        _NLEVELS -= 1

    def _int_handler(self, signum, frame):
        global _NLEVELS, _MAX_NLEVELS
        _NLEVELS = 0
        _MAX_NLEVELS = 0
        signal.signal(signal.SIGINT, self._signal_old)
        e = KeyboardInterrupt()
        e.__traceback__ = traceback.extract_stack(frame)
        raise e


def progress_bar(n, info=''):
    """
    Return a default progress bar.

    Example
    -------
    >>> import time
    >>> n = 10
    >>> bar = progress_bar(n, 'LOOP')
    >>> for i in range(n):
    ...     time.sleep(1)
    ...     bar.update()

    """
    return _ProgressBar(widgets=[info, Percentage(), Bar('=', '[', ']'),
                                 ETA()], maxval=n).start()


def _compress_mask(mask):
    mask = mask.ravel()
    if len(mask) == 0:
        return ''
    output = ''
    old = mask[0]
    n = 1
    for new in mask[1:]:
        if new is not old:
            if n > 2:
                output += str(n)
            elif n == 2:
                output += '+' if old else '-'
            output += '+' if old else '-'
            n = 1
            old = new
        else:
            n += 1
    if n > 2:
        output += str(n)
    elif n == 2:
        output += '+' if old else '-'
    output += '+' if old else '-'
    return output


def _uncompress_mask(mask):
    i = 0
    l = []
    nmask = len(mask)
    while i < nmask:
        val = mask[i]
        if val == '+':
            l.append(True)
            i += 1
        elif val == '-':
            l.append(False)
            i += 1
        else:
            j = i + 1
            val = mask[j]
            while val not in ('+', '-'):
                j += 1
                val = mask[j]
            l.extend(int(mask[i:j]) * (True if val == '+' else False,))
            i = j + 1
    return np.array(l, bool)


def create_folder_if_not_exists(folder_name):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        try:
            # Create the folder if it doesn't exist
            os.makedirs(folder_name)
            print(f"The folder '{folder_name}' has been created.")
        except OSError as e:
            print(f"Error creating the folder '{folder_name}': {e}")
    else:
        pass


def do_gif(input_folder, N, filename, output='animation.gif'):

    nmaps = np.arange(1, N+1, 1)
    image_list = []
    #for filename in sorted(os.listdir(input_folder)):
    for n in nmaps:
        image_path = os.path.join(input_folder, filename+f'{n}.png')
        image = Image.open(image_path)
        image_list.append(image)

    output_gif_path = os.path.join(input_folder, output)
    #output_gif_path = f"figures/{stk}/animation.gif"
    image_list[0].save(output_gif_path, save_all=True, append_images=image_list[1:], duration=100, loop=0)

def find_file(filename):
    '''
    find the full path to the file given the filename
    It could be a dictionary, or a data, or calibration file.

    We look first of all for the name, as given, which could be an absolute path
    We then look in the current working directory

    We also look in directories that have been defined in BASH environment variables

    Finally, we look in the package directory for dictionary, or data (including calfiles)

    we return the path name of the file that was found
    Otherwise we print a "not found" error, and return None
    '''

    if os.path.isfile(filename):
        return filename

    basename = os.path.basename(filename)
    dir_list = ['.']

    if 'QUBIC_DICT' in os.environ.keys():
        dir_list.append(os.environ['QUBIC_DICT'])

    if 'QUBIC_DATADIR' in os.environ.keys():
        dir_list.append(os.environ['QUBIC_DATADIR'])
        
    dir_list += [dicts_dir,cal_dir,data_dir]
    
    for D in dir_list:
        filename_fullpath = os.path.join(D,basename)
        if os.path.isfile(filename_fullpath):
            return filename_fullpath

    # if we get this far, then we haven't found the file
    print('ERROR!  File not found: %s' % basename)
    print('        I looked in the following directories:\n    %s' % '     \n'.join(dir_list))
    return None
