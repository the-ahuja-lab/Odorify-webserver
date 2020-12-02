from uuid import uuid4
from django.http import request
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
import os
import shutil
from .models import queuedisp,result,disp,disp4,disp2,disp3
from zipfile import ZipFile 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import traceback
import uuid
root = os.path.abspath('./')
def get_id(request):
	if 'id' in request.session:
		return str(request.session['id'])
	else:
		id = uuid.uuid4()
		request.session['id'] = id.int
		return str(id.int)

def fastaformat(s):
	t = []
	temp = ""
	for i in s[1:]:	
		if i.startswith(">"):
			t.append(temp)
			temp = ""
			continue	
		temp = temp+i
	t.append(temp)
	return t
def writeresult(a,job_name):
	temp = f'{job_name}/temp.txt'
	result = f"{job_name}/result.txt"
	with open(result,'r') as f:
		with open(temp,'w') as f2: 
			f2.write(f"{a.job_name}\n")
			f2.write(f"{a.count}\n")
			f2.write(f"{a.model}\n")
			f2.write(f"{a.id}\n")
			f2.write(f.read())
	try:
		os.rename(temp, result)
	except WindowsError:
		os.remove(result)
		os.rename(temp, result)

def readresult(user):
	with open(f"{user}/result.txt",'r') as f: # the current directory is generated
		s = result()
		s.job_name = f.readline().replace("\n","")
		if len(s.job_name) == 0:
			return
		else:
			s.count = int(f.readline().replace("\n",""))
			s.model = int(f.readline().replace("\n",""))
			s.id = (f.readline().replace("\n",""))
			return s		

def m2_file(job_name,k):
	data = pd.read_csv("output.csv")
	index = data.index
	number_of_rows = len(data)
	os.mkdir(f"{job_name}/{k}")
	for i in range(number_of_rows):
		if pd.isna(data["Probability"][i]):
			break
		shutil.move(f"{i+1}_SmileInterpretability.png", f"{job_name}/{k}")
		shutil.move(f"{i+1}_SequenceInterpretability.png", f"{job_name}/{k}")
		shutil.move(f"{i+1}_mol.svg", f"{job_name}/{k}")
	shutil.move("output.csv",f"{job_name}/{k}")
	os.remove("temp.csv")

def check_user(request):
	os.chdir(root)
	id = get_id(request)
	generated = "olfy/static/olfy/generated"
	if not os.path.isdir(f"{generated}/{id}"): 
		os.makedirs(f"{generated}/{id}/m1")
		os.makedirs(f"{generated}/{id}/m2")
		os.makedirs(f"{generated}/{id}/m3")
		os.makedirs(f"{generated}/{id}/m4")	
		f = open(f"{generated}/{id}/result.txt", 'w')
		f.close()
	return id

def m3_file(job_name,k):
	data = pd.read_csv("output.csv")
	index = data.index
	number_of_rows = len(data)
	os.mkdir(f"{job_name}/{k}")
	for i in range(number_of_rows):
		if pd.isna(data["Probability"][i]):
			break
		shutil.move(f"{i+1}_SmileInterpretability.png", f"{job_name}/{k}")
		shutil.move(f"{i+1}_SequenceInterpretability.png", f"{job_name}/{k}")
		shutil.move(f"{i+1}_mol.svg", f"{job_name}/{k}")
	shutil.move("output.csv",f"{job_name}/{k}")
	os.remove("temp.csv")

def move_files(job_name):
	shutil.move("mol.svg", job_name)
	shutil.move("lrp.pdf", job_name)
	os.remove("map.txt")

def loadingpage(request):
	os.chdir(root)
	return render(request, "olfy/Ahuja labs website/loading.html", {'hide': 'd-none'})

def home(request):
	if "GET" == request.method:
		check_user(request)	
		os.chdir(root)
		context= {
			'hide': 'd-none'
		}
		return render(request, "olfy/Ahuja labs website/index.html", context)

def displaymodels(request):
	check_user(request)	
	return render(request, "olfy/Ahuja labs website/modelsList.html")

def about(request):
	os.chdir(root)
	context= {
		'team': [
			{
				'name': 'Dr. Gaurav Ahuja',
				'post': 'Principal Investigator',
				'email': 'gaurav.ahuja@iiitd.ac.in'
			},
			{
				'name': 'Vishesh Agarwal',
				'post': 'Deep Learning & Interpretability',
				'email': 'vishesh18420@iiitd.ac.in'
			},
			{
				'name': 'Ria Gupta',
				'post': 'Deep Learning & Interpretability',
				'email': 'ria18405@iiitd.ac.in'
			},
			{
				'name': 'Sushant Gupta',
				'post': 'Back-End Development ',
				'email': 'sushant19450@iiitd.ac.in'
			},
			{
				'name': 'Rishi Raj Jain',
				'post': 'Front-End Development',
				'email': 'rishi18304@iiitd.ac.in'
			},
			{
				'name': 'Aayushi Mittal',
				'post': 'Data Collection',
				'email': 'aayushim@iiitd.ac.in'
			},
			{
				'name': 'Riya Sogani',
				'post': 'General web development',
				'email': 'riya19442@iiitd.ac.in'
			}
		]
	}
	return render(request, "olfy/Ahuja labs website/about.html", context)


def help(request):
	os.chdir(root)
	check_user(request)	
	context={
		'tabledata': [
			{
				'name': 'Odorant Predictor',
				'columnames': 'smiles (required in lowercase)',
				'extension': '*.csv'
			},
			{
				'name': 'OR Finder',
				'columnames': 'smiles (required in lowercase)',
				'extension': '*.csv'
			},
			{
				'name': 'Odor Finder',
				'columnames': 'seq (required in lowercase)',
				'extension': '*.csv'
			},
			{
				'name': 'Odorant-OR Pair Analysis',
				'columnames': 'smiles (required in lowercase), seq (required in lowercase)',
				'extension': '*.csv'
			}
		],
		'troubleshoot':  [
			{
				'ques': 'How can I integrate your product with PayPal?',
				'ans': 'Simple, you cannot.'
			}
		]
	}
	return render(request, "olfy/Ahuja labs website/help.html", context)

def results(request):
	if request.method == "GET":
		os.chdir(root)
		id = check_user(request)
		user = f"olfy/static/olfy/generated/{id}"
		a = readresult(user)
		if a is None: 
			return render(request, "olfy/Ahuja labs website/results.html",{"z": False})
		elif a.model == 1:
			s = f'{user}/m1/{a.job_name}/predicted_output.csv'
			data = pd.read_csv(s)
			index = data.index
			number_of_rows = len(index)
			display = []
			for i in range(number_of_rows):
				b = disp()
				b.smiles = data["smiles"][i]
				b.prob = data["prob"][i]
				b.sno = i+1
				temp = data["pred_odor"][i]
				if temp == 1:
					odor = "odorant"
				else:
					odor = "odorless"
				b.odor = odor
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result": a, "z":True, "display": [{"row": display}], "singleT": True})
		elif a.model == 2:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m2/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["smiles"])
				temp = {}
				temp["smiles"] = data["smiles"][0]
				temp1 = []
				for j in range(number_of_rows):
					b = disp2()
					b.sno = j+1
					if pd.isna(data["Final_Sequence"][j]):
						b.seq = "NA"
						b.receptorname = "NA"
						b.prob = "NA"
						b.noresult = True
					else:						
						b.seq = data["Final_Sequence"][j]
						b.receptorname = data["Receptor"][j]
						b.prob = data["Probability"][i]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display, "singleT": None})
		elif a.model == 3:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m3/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["Smiles"])
				temp = {}
				temp["seq"] = (data["seq"][0])
				temp1 = []
				for j in range(number_of_rows):
					b = disp3()
					b.sno = j+1
					if pd.isna(data["Probability"][j]):
						b.smiles = "NA"
						b.prob = "NA"
						b.noresult = True
					else:						
						b.smiles = data["Smiles"][j]
						b.prob = data["Probability"][i]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display, "singleT": None})
		elif a.model == 4:
			s = f'{user}/m4/{a.job_name}/output.csv'
			data = pd.read_csv(s)
			index = data.index
			number_of_rows = len(index)
			display = []
			for i in range(number_of_rows):
				b = disp4()
				b.smiles = data["smiles"][i]
				b.prob = data["prob"][i]
				b.sno = i+1
				b.seq = data["seq"][i]
				if data["status"][i] == 0:
					b.status = "non binding"
				else:
					b.status = "binding"
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True, "display": [{"row": display}], "singleT": True})

def result_queue(request,job_name,model,count):
	if request.method == "GET":
		os.chdir(root)
		id = check_user(request)
		user = f"olfy/static/olfy/generated/{id}"
		a = result()
		a.job_name = job_name
		a.model = int(model)
		a.count = int(count)
		a.id = id
		if a is None: 
			return render(request, "olfy/Ahuja labs website/results.html",{"z": False})
		elif a.model == 1:
			s = f'{user}/m1/{a.job_name}/predicted_output.csv'
			data = pd.read_csv(s)
			index = data.index
			number_of_rows = len(index)
			display = []
			for i in range(number_of_rows):
				b = disp()
				b.smiles = data["smiles"][i]
				b.prob = data["prob"][i]
				b.sno = i+1
				b.smile_id = i
				temp = data["pred_odor"][i]
				if temp == 1:
					odor = "odorant"
				else:
					odor = "odorless"
				b.odor = odor
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result": a, "z":True, "display": [{"row": display}], "singleT": True})
		elif a.model == 2:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m2/{a.job_name}/{i+1}/output.csv')

				number_of_rows = len(data["smiles"])
				temp = {}
				temp["smiles"] = (data["smiles"][0])
				temp1 = []
				for j in range(number_of_rows):
					b = disp2()
					b.sno = j+1
					b.seq = data["Final_Sequence"][j]
					b.receptorname = data["Receptor"][j]
					b.link = data["ensemble_link"]
					b.gene = data["Gene stable ID"]
					b.prob = data["Probability"][i]
					if "threshhold" in data.columns:
						b.threshhold = data["threshhold"][0]
						b.rapid = True
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display, "singleT": None})
		elif a.model == 3:
			display = []
			for i in range(a.count):
				data = pd.read_csv(f'{user}/m3/{a.job_name}/{i+1}/output.csv')
				number_of_rows = len(data["Smiles"])
				temp = {}
				temp["seq"] = (data["seq"][0])
				temp1 = []
				for j in range(number_of_rows):
					b = disp3()
					b.sno = j+1
					b.smiles = data["Smiles"][j]
					b.prob = data["Probability"][i]
					b.tableno = i+1
					temp1.append(b)
				temp["row"] = temp1
				display.append(temp)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True,"display":display, "singleT": None})
		elif a.model == 4:
			s = f'{user}/m4/{a.job_name}/output.csv'
			data = pd.read_csv(s)
			index = data.index
			number_of_rows = len(index)
			display = []
			for i in range(number_of_rows):
				b = disp4()
				b.smiles = data["smiles"][i]
				b.prob = data["prob"][i]
				b.sno = i+1
				b.seq = data["seq"][i]
				if data["status"][i] == 0:
					b.status = "non binding"
				else:
					b.status = "binding"
				display.append(b)
			return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":True, "display": [{"row": display}], "singleT": True})
		return render(request, "olfy/Ahuja labs website/results.html",{"result":a,"z":False})

def odor(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)		
		return render(request, "olfy/Ahuja labs website/odor.html")
	else:
		try:
			os.chdir(root)
			a = result()
			id = check_user(request)
			a.id = id
			userm1 = f"../{id}/m1"
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			smiles = request.POST["smiles"]
			email = request.POST["email"]
			m1 = f"olfy/static/olfy/generated/m1"
			os.chdir(m1)
			s = smiles.replace('\r',"").split('\n')
			if "" in s:
				s.remove("")
			temp = {"smiles":s}
			data = pd.DataFrame(temp)
			data = data.head(25)
			data.to_csv("input.csv",index=False)
			os.system("python transformer-cnn.py config.cfg")
			count = 1
			while os.path.isdir(f"{userm1}/{job_name}"):
				job_name = f"{job_name}1"
			os.mkdir(f"{userm1}/{job_name}")
			a.job_name = job_name
			job_name = f"{userm1}/{job_name}"
			a.model = 1
			f = pd.read_csv("input.csv")
			smiles = f["smiles"]
			a.count = len(smiles)
			for i in smiles:
				s = "python ochem.py detectodor.pickle "+ f'"{i}"'
				os.system(s)
				os.system("gnuplot map.txt && python generate_table.py")
				smile_path = f"{job_name}/{count}"
				os.makedirs(smile_path)			
				move_files(smile_path)
				count+=1
			os.remove("results.csv")
			shutil.move("predicted_output.csv", job_name)
			os.remove("input.csv")
			a.count = count
			os.chdir("../")
			writeresult(a,id)
			os.chdir("../../../../")
			if len(email)!=0:
				send_attachment(a,email,request)
			return JsonResponse({'code': 1})
		except Exception as e:
			traceback.print_exc()
			os.chdir(root)
			return JsonResponse({'code': 0})

# def getEmail(request):
# 	if "POST" == request.method:
# 		os.chdir(root)
# 		return JsonResponse({'code': 1})
# 		# registerEmail(request.POST['email'])

def odor_Or(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)			
		return render(request, "olfy/Ahuja labs website/odorOR.html")
	else:
		try:
			os.chdir(root)			
			a = result()
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			smiles = request.POST["smiles"]
			fasta = request.POST["fasta"]
			email = request.POST["email"]
			id = check_user(request)
			a.id = id
			m4 = "olfy/static/olfy/generated/m4"
			os.chdir(m4)
			s = smiles.replace('\r',"").split('\n')
			if "" in s:
				s.remove("")
			t = fasta.replace('\r',"").split('\n')
			if "" in t:
				t.remove("")
			temp = {"smiles":s,"seq":t}
			data = pd.DataFrame(temp)
			data = data.head(25)
			data.to_csv("input.csv",index=False)
			userm4 = f"../{id}/m4"
			os.system("python M4_final.py")
			while os.path.isdir(f"{userm4}/{job_name}"):
				job_name = f"{job_name}1"
			a.job_name = job_name
			job_name = f"{userm4}/{job_name}"
			os.mkdir(job_name)
			a.model = 4
			data = pd.read_csv("output.csv")
			output = []
			index = data.index
			number_of_rows = len(index)
			for i in range(number_of_rows):
				shutil.move(f"{i+1}_SmileInterpretability.png", job_name)
				shutil.move(f"{i+1}_SequenceInterpretability.png", job_name)
				shutil.move(f"{i+1}_mol.svg", job_name)
			shutil.move("output.csv",job_name)
			os.remove("input.csv")
			a.count = number_of_rows
			os.chdir("../")
			writeresult(a,id)
			for i in range(4):
				os.chdir("../")
			if len(email)!=0:
				send_attachment(a,email,request)			
			return JsonResponse({'code': 1})
		except Exception as e:
			os.chdir(root)
			traceback.print_exc()
			return JsonResponse({'code': 0})	

def Or(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)	
		return render(request, "olfy/Ahuja labs website/or.html")		
	else:
		try:
			os.chdir(root)
			a = result()
			id = check_user(request)
			a.id = id
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			fasta = request.POST["fasta"]
			email = request.POST["email"]
			counter = request.POST["normal_counter"]
			m3 = "olfy/static/olfy/generated/m3"
			os.chdir(m3)
			t = fasta.replace('\r',"").split('\n')
			if "" in t:
				t.remove("")
			t = fastaformat(t)
			temp = {"seq":t}
			data = pd.DataFrame(temp)
			data = data.head(25)
			data.to_csv("input.csv",index=False)
			a.model = 3
			while os.path.isdir(f"../{id}/m3/{job_name}"):
				job_name = f"{job_name}1"
			a.job_name = job_name
			job_name = f"../{id}/m3/{job_name}"
			os.mkdir(job_name)
			f = pd.read_csv("input.csv")
			a.count = len(f["seq"])
			for i in range(len(f["seq"])):
				dic = {"seq":[f["seq"][i]],"k":int(counter)}
				df = pd.DataFrame(dic)
				df.to_csv("temp.csv",index=False)
				os.system("python M3.py")
				df = pd.read_csv("output.csv")
				j = []
				for k in range(len(df["Probability"])):
					j.append(f["seq"][i])
				df["seq"] = j
				df.to_csv("output.csv",index=False)
				m3_file(job_name,i+1);
			os.remove("input.csv")
			os.chdir("../")
			writeresult(a,id)
			for i in range(4):
				os.chdir("../")
			if len(email)!=0:
				send_attachment(a,email,request)
			return JsonResponse({'code': 1})
		except Exception as e:
			traceback.print_exc()
			os.chdir(root)
			return JsonResponse({'code': 0})	
		
def odor2(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)	
		return render(request, "olfy/Ahuja labs website/odor2.html")
	else:
		try:
			os.chdir(root)
			a = result()
			id = check_user(request)
			a.id = id
			job_name = request.POST["job_name"]
			if len(job_name) == 0:
				job_name = "untitled"
			smiles = request.POST["smiles"]
			email = request.POST["email"]
			slider = request.POST["slider_value"]
			counter = request.POST["normal_counter"]
			m2 = "olfy/static/olfy/generated/m2"
			os.chdir(m2)
			t = smiles.replace('\r',"").split('\n')
			if "" in t:
				t.remove("")
			temp = {"smiles":t}
			data = pd.DataFrame(temp)
			data = data.head(25)
			data.to_csv("input.csv",index=False)
			a.model = 2
			userm2 = f"../{id}/m2"
			while os.path.isdir(f"{userm2}/{job_name}"):
				job_name = f"{job_name}1"
			a.job_name = job_name
			job_name = f"{userm2}/{job_name}"
			os.mkdir(job_name)
			f = pd.read_csv("input.csv")
			a.count = len(f["smiles"])
			for i in range(len(f["smiles"])):
				if counter == "10" and slider != "1":
					dic = {"smiles":[f["smiles"][i]],"threshhold":float(slider)}
					df = pd.DataFrame(dic)
					df.to_csv("temp.csv",index=False)
					os.system("python M2.py")
				elif slider == "1" and counter != "10":
					dic = {"smiles":[f["smiles"][i]],"k":int(counter)}
					df = pd.DataFrame(dic)
					df.to_csv("temp.csv",index=False)
					os.system("python M2-brute-force.py")
				else:
					dic = {"smiles":[f["smiles"][i]],"threshhold":float(slider)}
					df = pd.DataFrame(dic)
					df.to_csv("temp.csv",index=False)
					os.system("python M2.py")
				df = pd.read_csv("output.csv")
				j = []
				for k in range(len(df["Probability"])):
					j.append(f["smiles"][i])
				df["smiles"] = j
				# df=pd.merge(df, data, on='Receptor')
				df.to_csv("output.csv",index=False)
				m2_file(job_name,i+1);
			os.remove("input.csv")
			os.chdir("../")
			writeresult(a,id)
			for i in range(4):
				os.chdir("../")
			if len(email)!=0:
				send_attachment(a,email,request)			
			return JsonResponse({'code': 1})
		except Exception as e:
			traceback.print_exc()
			os.chdir(root)
			return JsonResponse({'code': 0})


def contactus(request):
	if "GET" == request.method:
		os.chdir(root)
		check_user(request)	
		return render(request, "olfy/Ahuja labs website/contact.html")
	else:
		os.chdir(root)
		email = request.POST["email"]
		subject = request.POST["title"]
		message = request.POST["message"]
		sender = "odorify.ahujalab@iiitd.ac.in"
		msg = MIMEMultipart()
		msg['From'] = sender
		msg['To'] = sender
		msg['Subject'] = subject
		msg.attach(MIMEText(message, 'plain'))
		text = msg.as_string() 
		s = smtplib.SMTP('smtp.gmail.com', 587)
		s.starttls()
		s.login(sender, "odorify123")
		s.sendmail(sender, sender, text)
		msg = MIMEMultipart()
		msg['From'] = sender
		msg['To'] = email
		msg['Subject'] = "Thank You for your response"
		message = "Dear User,\nThank you for your response. We will try to contact you as soon as possible"
		msg.attach(MIMEText(message, 'plain'))
		text = msg.as_string() 
		s.sendmail(sender, email, text)
		s.quit()
		return render(request, "olfy/Ahuja labs website/contact.html", {'messages': ['✓ Mail sent successfully']})

def queue(request):
	if "GET" == request.method:
		os.chdir(root)
		id = check_user(request)
		f = open(f"olfy/static/olfy/generated/{id}/result.txt")
		data = f.read().splitlines()
		length = len(data)
		queue = []
		for i in range(0,length,4):
			temp = queuedisp()
			temp.count = data[i+1]
			temp.job_name = data[i]
			temp.sno = (i//4)+1
			temp.model = data[i+2]
			if temp.model == '1':
				temp.model_name = "Odorant Predictor"
			elif temp.model == '2':
				temp.model_name = "OR Finder"
			elif temp.model == '3':
				temp.model_name = "Odor Finder"
			elif temp.model == '4':
				temp.model_name = "Odorant-OR Pair Analysis"
			queue.append(temp)
		return render(request, "olfy/Ahuja labs website/queue.html",{"queue":queue})

def makezip(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m1")
	for i in range(a.count):
		file_path.append(f"{a.job_name}/{i}/lrp.pdf")
		file_path.append(f"{a.job_name}/{i}/mol.svg") 
	file_path.append(f"{a.job_name}/predicted_output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(5):
		os.chdir("../")
	for i in range(6):
		os.chdir("../")
	return zip

def makezip2(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m2")
	for i in range(a.count):
		f = pd.read_csv(f"{a.job_name}/{i+1}/output.csv")
		count = len(f["smiles"])
		for j in range(count):
			file_path.append(f"{a.job_name}/{i+1}/{i+1}_SmileInterpretability.png")
			file_path.append(f"{a.job_name}/{i+1}/{i+1}_SequenceInterpretability.png") 
			file_path.append(f"{a.job_name}/{i+1}/{i+1}_mol.svg") 
		file_path.append(f"{a.job_name}/{i+1}/output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")
	return zip

def makezip3(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m3")
	for i in range(a.count):
		f = pd.read_csv(f"{a.job_name}/{i+1}/output.csv")
		count = len(f["smiles"])
		for j in range(count):
			file_path.append(f"{a.job_name}/{i+1}/{i+1}_SmileInterpretability.png")
			file_path.append(f"{a.job_name}/{i+1}/{i+1}_SequenceInterpretability.png") 
			file_path.append(f"{a.job_name}/{i+1}/{i+1}_mol.svg") 
		file_path.append(f"{a.job_name}/{i+1}/output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")	
	return zip

def makezip4(a,request):
	os.chdir(root)
	id = check_user(request)
	file_path = []
	os.chdir(f"olfy/static/olfy/generated/{id}/m4")
	for i in range(a.count):
		file_path.append(f"{a.job_name}/{i+1}_SmileInterpretability.png")
		file_path.append(f"{a.job_name}/{i+1}_SequenceInterpretability.png") 
		file_path.append(f"{a.job_name}/{i+1}_mol.svg") 
	file_path.append(f"{a.job_name}/output.csv")
	zip = ZipFile(f"{a.job_name}/data.zip",'w') 
	for file in file_path: 
		zip.write(file)
	zip.close()
	zip = open(f"{a.job_name}/data.zip","rb")
	for i in range(6):
		os.chdir("../")
	return zip

def download(request,job_name,model,count):
	os.chdir(root)
	a = result()
	a.job_name = job_name
	a.model = int(model)
	a.count = int(count)
	zip = ''
	if a.model==1:
		zip = makezip(a,request)
	if a.model==2:
		zip = makezip2(a,request)
	if a.model==3:
		zip = makezip3(a,request)
	if a.model==4:
		zip = makezip4(a,request)
	response = HttpResponse(zip,content_type='application/zip')
	response['Content-Disposition'] = 'attachment; filename=data.zip'
	return response

def send_attachment(a,email,request):
	os.chdir(root)
	attachment = ""
	sender = "odorify.ahujalab@iiitd.ac.in"
	if a.model==1:
		attachment = makezip(a,request)
	if a.model==2:
		attachment = makezip2(a,request)
	if a.model==3:
		attachment = makezip3(a,request)
	if a.model==4:
		attachment = makezip4(a,request)

	msg = MIMEMultipart()
	msg['From'] = sender
	msg['To'] = email
	msg['Subject'] = "Results"
	filename = "data.zip"
	p = MIMEBase('application', 'octet-stream') 
	p.set_payload((attachment).read()) 
	encoders.encode_base64(p) 
	p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
	msg.attach(p) 
	message = "Dear User,\n Thank you for using Odorify. Please find attached your combined results in a zip file. In case of any queries, please contact us through the help page of our webserver."
	msg.attach(MIMEText(message, 'plain'))	
	text = msg.as_string() 
	s = smtplib.SMTP('smtp.gmail.com', 587)
	s.starttls()
	s.login(sender, "odorify123")
	s.sendmail(sender, email, text)
	s.quit()