<<<<<<< HEAD
<<<<<<< HEAD
import os,sys
=======
import os, sys
>>>>>>> master
import string


def ask_for(key):
    if sys.version_info.major == 2:
        s = raw_input("flipperDict: enter value for '%s': " % key)
    else:
<<<<<<< HEAD
        s = input( "flipperDict: enter value for '%s': " % key )
=======
import os, sys
import string


def ask_for(key):
    if sys.version_info.major == 2:
        s = raw_input("flipperDict: enter value for '%s': " % key)
    else:
        s = input("flipperDict: enter value for '%s': " % key)
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
        s = input("flipperDict: enter value for '%s': " % key)
>>>>>>> master
    try:
        val = eval(s)
    except NameError:
        # allow people to enter unquoted strings
        val = s
    return val

<<<<<<< HEAD
<<<<<<< HEAD
class qubicDict( dict ):
    # assign the directory where to find dictionaries by default
    dicts_dir = os.path.dirname(__file__) + '/dicts'

    def __init__( self, ask = False):
=======

class qubicDict(dict):
    # assign the directory where to find dictionaries by default
    dicts_dir = os.path.dirname(__file__) + '/dicts'

    def __init__(self, ask=False):
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======

class qubicDict(dict):
    # assign the directory where to find dictionaries by default
    dicts_dir = os.path.dirname(__file__) + '/dicts'

    def __init__(self, ask=False):
>>>>>>> master
        """
        @param ask if the dict doesn't have an entry for a key, ask for the associated value and assign
        """
        dict.__init__(self)
        self.ask = ask

        return

<<<<<<< HEAD
<<<<<<< HEAD
    def __getitem__( self, key ):
=======
    def __getitem__(self, key):
>>>>>>> master
        if key not in self:
            if self.ask:
                print("flipperDict: parameter '%s' not found" % key)
                val = ask_for(key)
                print("flipperDict: setting '%s' = %s" % (key, repr(val)))
                dict.__setitem__(self, key, val)
            else:
                return None
        return dict.__getitem__(self, key)

<<<<<<< HEAD
    def read_from_file( self, filename ):
=======
    def __getitem__(self, key):
        if key not in self:
            if self.ask:
                print("flipperDict: parameter '%s' not found" % key)
                val = ask_for(key)
                print("flipperDict: setting '%s' = %s" % (key, repr(val)))
                dict.__setitem__(self, key, val)
            else:
                return None
        return dict.__getitem__(self, key)

    def read_from_file(self, filename):
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
    def read_from_file(self, filename):
>>>>>>> master
        '''
        read a given dictionary file
        '''

        # read from default location if the filename is not found
        if not os.path.isfile(filename):
            basename = os.path.basename(filename)

            if 'QUBIC_DICT' in os.environ.keys():
                # read from the users QUBIC_DICT path if defined
<<<<<<< HEAD
<<<<<<< HEAD
                filename = os.environ['QUBIC_DICT']+os.sep+basename

            if not os.path.isfile(filename):
                # try to read from the package path
                filename = self.dicts_dir+os.sep+basename
=======
                filename = os.environ['QUBIC_DICT'] + os.sep + basename

            if not os.path.isfile(filename):
                # try to read from the package path
                filename = self.dicts_dir + os.sep + basename
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
                filename = os.environ['QUBIC_DICT'] + os.sep + basename

            if not os.path.isfile(filename):
                # try to read from the package path
                filename = self.dicts_dir + os.sep + basename
>>>>>>> master

                if not os.path.isfile(filename):
                    print('Could not read dictionary.  File not found: %s' % basename)
                    return
<<<<<<< HEAD
<<<<<<< HEAD
            
        f = open( filename )
=======

        f = open(filename)
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======

        f = open(filename)
>>>>>>> master
        old = ''
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            s = line.split('#')
            line = s[0]
            s = line.split('\\')
            if len(s) > 1:
                old = ' '.join([old, s[0]])
                continue
            else:
                line = ' '.join([old, s[0]])
                old = ''
            for i in range(len(line)):
<<<<<<< HEAD
<<<<<<< HEAD
                if line[i]!=' ':
=======
                if line[i] != ' ':
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
                if line[i] != ' ':
>>>>>>> master
                    line = line[i:]
                    break
            exec(line)
            s = line.split('=')
            if len(s) != 2:
                print("Error parsing line:")
                print(line)
                continue
            key = s[0].strip()
<<<<<<< HEAD
<<<<<<< HEAD
            val = eval(s[1].strip()) # XXX:make safer
=======
            val = eval(s[1].strip())  # XXX:make safer
>>>>>>> master
            self[key] = val
        f.close()
        # self.prefix_OutputName()

    readFromFile = read_from_file

    def write_to_file(self, filename, mode='w'):
        f = open(filename, mode)
        keys = self.keys()
        keys.sort()
        for key in keys:
<<<<<<< HEAD
            f.write( "%s = %s\n" % (key,repr(self[key])) )
=======
            val = eval(s[1].strip())  # XXX:make safer
            self[key] = val
        f.close()
        # self.prefix_OutputName()

    readFromFile = read_from_file

    def write_to_file(self, filename, mode='w'):
        f = open(filename, mode)
        keys = self.keys()
        keys.sort()
        for key in keys:
            f.write("%s = %s\n" % (key, repr(self[key])))
>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
            f.write("%s = %s\n" % (key, repr(self[key])))
>>>>>>> master
        f.close()

    writeToFile = write_to_file

<<<<<<< HEAD
<<<<<<< HEAD
    def cmp( self, otherDict ):
        
=======
    def cmp(self, otherDict):

>>>>>>> 817389f4cc3163541fa042c883a3919ba9169a19
=======
    def cmp(self, otherDict):

>>>>>>> master
        diff = []
        ks = self.keys()
        for k in ks:
            try:
                if otherDict[k] == self.params[k]:
                    continue
                diff += [k]
                break
            except KeyError:
                diff += [k]
        return otherDict

#    def prefix_OutputName(self):
#
#        import datetime
#        if not self['output']:
#            self['output']='./'
#        dir_output = str(self['output'])
#
#        if os.path.isdir( dir_output ):
#            print( 'QUBIC output directory: {}'.format( dir_output ) )  
#        elif not os.path.isdir( dir_output ):
#            print( 'Building output directory' )
#            os.mkdir( dir_output )
#            print( 'Built it. QUBIC output directory: {}'.format( dir_output ) )
#
#        now = datetime.datetime.now()
#        today = now.strftime( "%Y%m%d" )
#
#        files = os.listdir( dir_output )
#        new_v = []
#        last = "00"
#
#        for each in files:
#            each_cut = each[0:8]
#            if today == each_cut:
#                new_v.append(each[9:11])
#                new_v.sort()
#                last = str(int(new_v[-1])+1).zfill(2)
#
#        if dir_output[-1] == "/":
#            self['prefix'] = str(dir_output)+str(today)+"_"+str(last)+"_"
#        else:
#            self['prefix'] = str(dir_output)+"/"+str(today)+"_"+str(last)+"_"
#
