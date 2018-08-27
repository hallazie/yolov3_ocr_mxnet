img_path = 'data/imgs/'
json_path = 'data/jsons/'
BBOX_DICT = {'equip_seq':0, 'code':1, 'num':2, 'date':3, 'checkcode':4, 'buyer_name':5, 'buyer_tax_seq':6, 'encrypt0':7, 
		'encrypt1':8, 'encrypt2':9, 'encrypt3':10, 'merch':11, 'merch_unit':12, 'merch_num':13, 'merch_unit_price':14,
		'merch_price':15, 'merch_tax_rate':16, 'merch_tax':17, 'merch_price_total':18, 'merch_price_zh':19, 'merch_price_nm':20,
		'sell_name':21, 'sell_tax_seq':22, 'sell_addr_phone':23, 'sell_bank_phone':24, 'receip':25, 'recheck':26, 'grant':27}
BBOX_KEY_LIST = sorted(BBOX_DICT.keys())
BBOX_DICT_REVERSE = {v:k for k,v in zip(BBOX_DICT.keys(), BBOX_DICT.values())}
THRESHOLD = 0.8
WIDTH = 320
HEIGHT = 160
DOWNSAMPLE = 32
BATCH_SIZE = 2
