from django.db import models

class result:
	model:int
	job_name:str
	count:int
	ipaddr:str
	
class disp:
	sno:int
	smiles:str
	prob:str
	odor:str
	smile_id:str

class disp4:
	sno:int
	smiles:str
	prob:str
	status:str
	seq:str

class disp2:
	sno:int
	seq:str
	receptorname:str
	prob:str
	tableno:int
	threshhold:float
	k:int

class disp3:
	sno:int
	smiles:str
	prob:str
	tableno:int
 
class queuedisp:
	sno:int
	job_name:str
	model:str
	model_name:str
	count:str
	
