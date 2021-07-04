import mysql.connector as mysql
import numpy as np
import base64
import PIL
from PIL import Image as Image
import io
import re


'''
Jupyter Notebook functions
'''
def showURI(imageuri):
    imgstr = re.search(r'base64,(.*)', imageuri).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    display(im)

def showNP(a, fmt='png'):
    a = np.uint8(a)
    f = io.StringIO()
    display(PIL.Image.fromarray(a))



'''
Image functions
'''
def uriToIMG(imageuri):
    imgstr = re.search(r'base64,(.*)', imageuri).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    return Image.open(image_bytes)

def imageToNp(im):
    arr = np.array(im)[:,:,0]
    return arr

def uriToNP(imageuri):
    imgstr = re.search(r'base64,(.*)', imageuri).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    nparray = np.array(im)
    if len(nparray.shape) == 2:
        arr = np.array(im)[:,:]
    elif len(nparray.shape) == 3:
        arr = np.array(im)[:,:,0]
    return arr

def pilToURI(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    ret = u'data:image/png;base64,'+str(img_str)[2:-1:]
    return ret

'''
database functions
'''
def createProcessingLayer(db, batchid, inputlayerid, name, description):
    #returns the id of the new processing layer
    dbcursor = db.cursor()
    dbcursor.execute(f"INSERT INTO processing_layers(batchid, inputlayerid, name, description) VALUES('{batchid}','{inputlayerid}','{name}','{description}')")
    db.commit()
    return dbcursor.lastrowid

def createBatch(db, name, description):
    dbcursor = db.cursor()
    dbcursor.execute(f"INSERT INTO batches(name, description) VALUES('{name}','{description}')")
    db.commit()
    return dbcursor.lastrowid, name, description

def getBatches(db):
    dbcursor = db.cursor()
    dbcursor.execute(f"SELECT * FROM batches")
    return dbcursor.fetchall()

# def saveImage(db, batchid, layerid, time, position, velocity, rotation, image):
#     insertCmd = "INSERT into images(time, position, velocity, rotation, batchid, layerid, image) VALUES ('"+str(time)+"', '"+str(position)+"','"+str(velocity)+"','"+str(rotation)+"','"+str(batchid)+"','"+str(layerid)+"','%s');"
#     try:
#         db.cursor().execute(insertCmd, (image))
#         db.commit()
#     except Exception as e:
#         print(f"Exception: {e}")

def saveImage(db, batchid, layerid, time, position, velocity, rotation, image):
    insertCmd = "INSERT into images(time, position, velocity, rotation, batchid, layerid, image) VALUES (%s,%s,%s,%s,%s,%s,%s);"
    try:
        db.cursor().execute(insertCmd, (time, position, velocity, rotation, batchid, layerid, image))
        db.commit()
    except Exception as e:
        print(f"Exception: {e}")
