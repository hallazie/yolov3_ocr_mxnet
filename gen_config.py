#coding:utf-8

from PIL import Image

import random
import time
import json

keys = ['equip_seq', 'buyer_name', 'buyer_tax_seq', 'encrypt', 'merch', 'merch_unit', 'merch_num', 'merch_unit_price', 
		'merch_price', 'merch_tax_rate', 'merch_tax', 'merch_price_total', 'merch_price_zh', 'merch_price_nm', 'sell_name', 'sell_tax_seq', 
		'sell_addr_phone', 'sell_bank_phone', 'receip', 'recheck', 'grant']
addr_file = open('pool/addr.txt', 'r')
addr_pool = [e.strip() for e in addr_file.readlines()]
addr_file.close()
bank_file = open('pool/bank.txt', 'r')
bank_pool = [e.strip() for e in bank_file.readlines()]
bank_file.close()
name_file = open('pool/name.txt', 'r')
name_pool = [e.strip() for e in name_file.readlines()]
name_file.close()
comp_file = open('pool/company.txt', 'r')
comp_pool = [e.strip() for e in comp_file.readlines()]
comp_file.close()	
addr_rang = len(addr_pool)
bank_rang = len(bank_pool)
name_rang = len(name_pool)
comp_rang = len(comp_pool)

def gen():
	receip_list = []
	for i in range(100):
		receip_list.append(gen_single())
	with open('tmp.json', 'w') as jfile:
		json.dump(receip_list, jfile, ensure_ascii=False)
		jfile.write('\n')

def gen_single():
	receip = {}
	font_path = 'font/'
	num_font = 'mriamc.ttf'
	chr_font = 'simsun.ttc'
	receip['equip_seq'] = {'chars':rand_num_seq(12), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(162,165), 'rdshift':True, 'space':0}
	receip['code'] = {'chars':rand_num_seq(12), 'font':font_path+num_font, 'fontsize':18, 'pos':(1079,72), 'rdshift':True, 'space':0}
	receip['num'] = {'chars':rand_num_seq(8), 'font':font_path+num_font, 'fontsize':18, 'pos':(1079,108), 'rdshift':True, 'space':0}
	receip['date'] = {'chars':rand_date(), 'font':font_path+chr_font, 'fontsize':18, 'pos':(1079,144), 'rdshift':True, 'space':0}
	receip['checkcode'] = {'chars':rand_check_seq(5), 'font':font_path+num_font, 'fontsize':18, 'pos':(1079,180), 'rdshift':True, 'space':0}
	receip['buyer_name'] = {'chars':rand_comp(), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(264,210), 'rdshift':True, 'space':0}
	receip['buyer_tax_seq'] = {'chars':rand_seq(18), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(264,240), 'rdshift':True, 'space':0}
	receip['encrypt0'] = {'chars':random_encrypt(), 'font':font_path+'mriamc.ttf', 'fontsize':18, 'pos':(883,218), 'rdshift':True, 'space':0}
	receip['encrypt1'] = {'chars':random_encrypt(), 'font':font_path+'mriamc.ttf', 'fontsize':18, 'pos':(883,248), 'rdshift':True, 'space':0}
	receip['encrypt2'] = {'chars':random_encrypt(), 'font':font_path+'mriamc.ttf', 'fontsize':18, 'pos':(883,278), 'rdshift':True, 'space':0}
	receip['encrypt3'] = {'chars':random_encrypt(), 'font':font_path+'mriamc.ttf', 'fontsize':18, 'pos':(883,308), 'rdshift':True, 'space':0}
	receip['merch'] = {'chars':rand_comp(), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(73,377), 'rdshift':True, 'space':0}
	receip['merch_unit'] = {'chars':rand_unit(), 'font':font_path+chr_font, 'fontsize':18, 'pos':(569,377), 'rdshift':True, 'space':0}
	receip['merch_num'] = {'chars':rand_num(), 'font':font_path+num_font, 'fontsize':18, 'pos':(708,377), 'rdshift':True, 'space':0}
	receip['merch_unit_price'] = {'chars':rand_num(), 'font':font_path+num_font, 'fontsize':18, 'pos':(818,377), 'rdshift':True, 'space':0}
	receip['merch_price'] = {'chars':str(round(float(receip['merch_num']['chars'])*float(receip['merch_unit_price']['chars']), 2)), 'font':font_path+num_font, 'fontsize':18, 'pos':(993,377), 'rdshift':True, 'space':0}
	receip['merch_tax_rate'] = {'chars':str(random.randint(1,40))+'%', 'font':font_path+num_font, 'fontsize':18, 'pos':(1096,377), 'rdshift':True, 'space':0}
	receip['merch_tax'] = {'chars':str(round(float(receip['merch_price']['chars'])*0.01*(float(receip['merch_tax_rate']['chars'][:-1])), 2)), 'font':font_path+num_font, 'fontsize':18, 'pos':(1248,377), 'rdshift':True, 'space':0}
	receip['merch_price_total'] = {'chars':'￥'+str(receip['merch_price']['chars']), 'font':font_path+'STKAITI.TTF', 'fontsize':18, 'pos':(983,580), 'rdshift':True, 'space':0}
	receip['merch_price_zh'] = {'chars':rand_price_zh(), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(400,630), 'rdshift':True, 'space':0}
	receip['merch_price_nm'] = {'chars':'￥'+str(round(float(receip['merch_price']['chars'])+float(receip['merch_tax_rate']['chars'][:-1]),2)), 'font':font_path+'STKAITI.TTF', 'fontsize':18, 'pos':(1218,630), 'rdshift':True, 'space':0}
	receip['sell_name'] = {'chars':rand_comp(), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(265,666), 'rdshift':True, 'space':0}
	receip['sell_tax_seq'] = {'chars':rand_num_seq(18), 'font':font_path+num_font, 'fontsize':18, 'pos':(265,696), 'rdshift':True, 'space':0}
	receip['sell_addr_phone'] = {'chars':rand_addr()+' '+rand_num_seq(11), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(265,729), 'rdshift':True, 'space':0}
	receip['sell_bank_phone'] = {'chars':rand_bank()+' '+rand_num_seq(11), 'font':font_path+'simsun.ttc', 'fontsize':18, 'pos':(265,759), 'rdshift':True, 'space':0}
	receip['receip'] = {'chars':rand_name(), 'font':font_path+'simsun.ttc', 'fontsize':20, 'pos':(170,800), 'rdshift':True, 'space':0}
	receip['recheck'] = {'chars':rand_name(), 'font':font_path+'simsun.ttc', 'fontsize':20, 'pos':(480,800), 'rdshift':True, 'space':0}
	receip['grant'] = {'chars':rand_name(), 'font':font_path+'simsun.ttc', 'fontsize':20, 'pos':(812,800), 'rdshift':True, 'space':0}
	return receip

def rand_addr():
	return addr_pool[random.randint(0, addr_rang-1)]

def rand_bank():
	return bank_pool[random.randint(0, bank_rang-1)]

def rand_name():
	return name_pool[random.randint(0, name_rang-1)]

def rand_comp():
	return comp_pool[random.randint(0, comp_rang-1)]

def rand_num_seq(rang):
	return ''.join([str(random.randint(0,9)) for i in range(rang)])

def rand_seq(rang):
	pool = ['0','1','2','3','4','5','6','7','8','9','Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M']
	return ''.join([pool[random.randint(0, len(pool)-1)] for i in range(rang)])

def rand_unit():
	pool = ['次','个','升','公斤','套','吨']
	return pool[random.randint(0, len(pool)-1)]

def rand_num():
	if random.randint(0,1):
		return str(random.randint(1,100))
	else:
		return str(random.randint(1,100)+round(random.random(), random.randint(1,5)))

def rand_date():
	return '20'+str(random.randint(1,18)).zfill(2)+'年'+str(random.randint(1,12)).zfill(2)+'月'+str(random.randint(1,30)).zfill(2)+'日'

def rand_check_seq(l):
	return rand_num_seq(5)+' '+rand_num_seq(5)+' '+rand_num_seq(5)+' '+rand_num_seq(5)

def rand_price_zh():
	pool = ['壹','拾','贰','叁','仟','肆','佰','伍','陆','万','柒','仟','捌','佰','玖']
	return ''.join([pool[random.randint(0, len(pool)-1)] for i in range(random.randint(5,10))])

def random_encrypt():
	pool = ['0','1','2','3','4','5','6','7','8','9','>','-','+','*','/']
	return ''.join([pool[random.randint(0, len(pool)-1)] for i in range(28)])

if __name__ == '__main__':
	gen()
