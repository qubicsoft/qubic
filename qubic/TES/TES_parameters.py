'''
$Id: TES_parameters.py
$auth: Steve Torchinsky <satorchi@apc.in2p3.fr>
$created: Thu 08 Feb 2018 07:43:48 CET
$license: GPLv3 or later, see https://www.gnu.org/licenses/gpl-3.0.txt

          This is free software: you are free to change and
          redistribute it.  There is NO WARRANTY, to the extent
          permitted by law.

Read a table which is a list of dictionary values giving the various parameters for each TES detector
Details of each parameter can be found in the text file, for example: QUBIC_TES_parameters.v0.3.txt


example:
 
   from qubic.TES.TES_parameters import *
   filename='QUBIC_TES_parameters.v0.3.txt'
   TESparameters_dictionary_list=TEStable_readfile(filename) 

'''
def TEStable_readfile(filename):
    '''
    read the text format table of TES parameters
    '''
    if not isinstance(filename,str) or not os.path.exists(filename):
        print('ERROR! File not found: %s' % str(filename))
        return None

    TEStable=[]

    h=open(filename,'r')
    contents=h.read()
    h.close()
    lines=contents.split('\n')

    # ignore all lines starting with '#'
    valid_lines=[]
    for line in lines:
        if not line.find('#')==0 and not line=='':
            valid_lines.append(line)

    # the first line should identify this as a valid file
    line=valid_lines[0]
    keyword,val=line.split('=')
    if not keyword=='FILEID':
        print('ERROR! This does not appear to be a valid QUBIC TES parameters table')
        return None
    if not val=='QUBIC table of TES parameters':
        print('ERROR! This does not appear to be a valid QUBIC TES parameters table')
        return None
    

    # second line should tell us the keywords
    line=valid_lines[1]
    keyword,val=line.split('=')
    if not keyword=='KEYWORDS':
        print('ERROR! Missing the keyword list!')
        return None
    keywords=val.split(';')
    
    msg='Expect to find the following keywords: '
    for keyword in keywords:
        msg+=keyword+' '
    print(msg+'\n')

    val_keywords=['INDEX','R300', 'ASIC', 'PIX', 'TES', 'NEP','G','T0','G','K','n']
    str_keywords=['OpenLoop', 'CarbonFibre', 'IV', 'DET_NAME']
    
    # process the rest of the file
    del(valid_lines[0])
    del(valid_lines[0])
    idx_counter=-1
    entry={}
    for line in valid_lines:
        keyword,str_val=line.split('=')
        if keyword in str_keywords or str_val=='NaN':
            val=str_val
        else:
            val=eval(str_val)
            
        if keyword=='INDEX':
            TEStable.append(entry) # store the previous entry
            entry={}
            idx=val
            idx_counter+=1
            if idx!=idx_counter:
                print('Weirdness: incompatible indeces.  INDEX=%i, idx_counter=%i' % (idx,idx_counter))

        entry[keyword]=val

    # add the last one
    TEStable.append(entry)        
    # clean up:  the first entry is empty because of the way we did the loop
    del(TEStable[0])
    
    
    return TEStable
