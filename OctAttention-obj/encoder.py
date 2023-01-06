'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from numpy import mod
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload,CPrintl,expName,device
from octAttention import model
import glob,datetime,os
import pt as pointCloud
############## warning ###############
## decoder.py relys on this model here
## do not move this lines to somewhere else
model = model.to(device)
saveDic = reload(None,'output/obj/checkpoint/encoder_epoch_00800110.pth')
model.load_state_dict(saveDic['encoder'])

###########Objct##############
list_orifile = []

def addFiles(dir):
    filenames=os.listdir(dir)
    print(filenames)
    for f in filenames:
        list_orifile.append(os.path.join(dir,f))


if __name__=="__main__":
    printl = CPrintl(expName+'/encoderPLY.txt')
    printl('_'*50,'OctAttention V0.4','_'*50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])
    addFiles('train/loot/Ply/')
    addFiles('train/redandblack/Ply/')
    for oriFile in list_orifile:
        printl(oriFile)
        if (os.path.getsize(oriFile)>300*(1024**2)):#300M
            printl('too large!')
            continue
        ptName = os.path.splitext(os.path.basename(oriFile))[0] 
        for qs in [1]:
            ptNamePrefix = ptName
            matFile,DQpt,refPt = dataPrepare(oriFile,saveMatDir='./output/obj/data',qs=qs,ptNamePrefix='',rotation=False)
            # please set `rotation=True` in the `dataPrepare` function when processing MVUB data
            main(matFile,model,actualcode=True,printl =printl) # actualcode=False: bin file will not be generated
            print('_'*50,'pc_error','_'*50)
            pointCloud.pcerror(refPt,DQpt,None,'-r 1023',None).wait()
