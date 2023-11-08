import sys
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader

def create_new_p4(stage_num,type,ternary_idx,table_num):
	RANGE_BIT=1
	TERNARY1Bit=2
	TERNARY4Bit=4
	TERNARY8Bit=8
	TERNARY16Bit=16
	#template_key={  'meta.prev_node_id': 'exact', \
    #        'meta.threshold_flag': 'exact', \
    #        'hdr.ipv4.total_len':'range', \
    #        'hdr.ipv4.protocol':'range', \
    #        'hdr.ipv4.flags[1:1]':'range', \
    #        'hdr.ipv4.flags[2:2]':'range', \
    #        'hdr.ipv4.ihl':'range', \
    #        'hdr.ipv4.ttl':'range', \
    #        'meta.srcPort':'range', \
    #        'meta.dstPort':'range', \
    #        'meta.flag[3:3]':'range', \
    #        'meta.flag[4:4]':'range', \
    #        'meta.flag[5:5]':'range', \
    #        'meta.flag[6:6]':'range', \
    #        'meta.flag[7:7]':'range' \
	#	  }
	#if ternary_type!= RANGE_BIT:
	#	template_key['meta.flag[1:1]']='ternary'
	#	template_key['meta.flag[2:2]']='ternary'
	#	template_key['meta.flag[3:3]']='ternary'
	#	template_key['meta.flag[4:4]']='ternary'
	#	template_key['meta.flag[5:5]']='ternary'
	#	template_key['meta.flag[6:6]']='ternary'
	#	template_key['meta.flag[7:7]']='ternary'
	#	
	#if ternary_type == TERNARY4Bit:
	#	print('comehereTERNARY4Bit')
	#	template_key['hdr.ipv4.ihl']='ternary'
	#elif ternary_type == TERNARY8Bit:
	#	print('comehereTERNARY8Bit')
	#	template_key['hdr.ipv4.ihl']='ternary'
	#	template_key['hdr.ipv4.protocol']='ternary'
	#	template_key['hdr.ipv4.ttl']='ternary'
	#elif ternary_type == TERNARY16Bit:
	#	print('comehereTERNARY16Bit')
	#	template_key['hdr.ipv4.total_len']='ternary'
	#	template_key['hdr.ipv4.protocol']='ternary'
	#	template_key['hdr.ipv4.ihl']='ternary'
	#	template_key['hdr.ipv4.ttl']='ternary'
	#	template_key['meta.srcPort']='ternary'
	#	template_key['meta.dstPort']='ternary'
	#print(template_key)
	#with open(pre_file) as f:
	#	template_str = f.read()
	#	template = Environment(loader=FileSystemLoader("./")).from_string(template_str)
	#	result = template.render(template_key=template_key) 	
	#p4_filepath = './'+'base.p4' 
	#with open(p4_filepath,'w') as f2:
	#	f2.write(result)	  
	FEATURE_BITS = [16, 8, 1, 8, 16, 16, 1, 1]
	match_name = [ 'range' if bits > ternary_idx else 'ternary'for bits in FEATURE_BITS]
	if type==0:
			file='./normal_tmplate.p4'
	elif type==1:
			file='./resubmit_tmplate.p4'
	else:
		file='./recirculate_tmplate.p4'
	print(file)
	levels=[]
	for i in range(stage_num):
		level_tmp='level'+'%d' % i
		levels.append(level_tmp)
	with open(file) as f:
		template_str = f.read()
		template = Environment(loader=FileSystemLoader("./")).from_string(template_str)
		result = template.render(levels=levels,match_name=match_name,table_num=table_num) 
	result.encode(encoding='utf-8')
	p4_filepath = './'+'create_new.p4' 
	with open(p4_filepath,'w') as f2:
		f2.write(result)


if __name__ == '__main__': 
	stages = sys.argv[1]
	type = sys.argv[2]
	ternary_type = sys.argv[3]
	table_num = sys.argv[4]
	stage_num=int(stages)
	type=int(type)
	ternary_type=int(ternary_type)
	create_new_p4(stage_num,type,ternary_type,table_num)

		