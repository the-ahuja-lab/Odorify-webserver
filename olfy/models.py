from django.db import models

class result:
	model:int
	job_name:str
	count:int
	id:str
	
class disp:
	sno:int
	smiles:str
	prob:str
	odor:str

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
	noresult = False

class disp3:
	sno:int
	smiles:str
	prob:str
	tableno:int
	noresult = False
 
class queuedisp:
	sno:int
	job_name:str
	model:str
	model_name:str
	count:str
	
