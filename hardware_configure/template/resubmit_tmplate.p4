{% extends "base.p4" %}


{% block header_resubmit_extra_info %}
header resubmit_h {
    bit<8>          threshold_flag;
    bit<16>          prev_node_id;
    bit<40>          pad;
}
{% endblock %}


{% block meta_resubmit_extra_info %}
resubmit_h      resubmit_data;
{% endblock %}


{% block start_parse %}
     state start {
        pkt.extract(ig_intr_md);
        transition select(ig_intr_md.resubmit_flag){
            1: parse_resubmit;
            0: parse_port_metadata;
        }
    }
	
    state parse_resubmit {
        pkt.extract(meta.resubmit_data);
        meta.threshold_flag = meta.resubmit_data.threshold_flag;
        meta.prev_node_id = meta.resubmit_data.prev_node_id;
        transition parse_ethernet;
    }
   state parse_port_metadata {
	   pkt.advance(PORT_METADATA_SIZE);
       meta.threshold_flag = 0;
       meta.prev_node_id = 0;
       transition parse_ethernet;
   }
{% endblock %}


{% block content %}
    action action_packet_add_info() {
		meta.resubmit_data.prev_node_id = meta.prev_node_id;
		meta.resubmit_data.threshold_flag = meta.threshold_flag;
		ig_dprsr_md.resubmit_type = 2;
   }
{% endblock %}

{% block content2 %}
	if(ig_intr_md.resubmit_flag==0)
	{
		action_packet_add_info();
	}
{% endblock %}


{% block resubmit_content %}
Resubmit() resubmit;
{% endblock %}


{% block content3 %}
	if (ig_dprsr_md.resubmit_type == 2) {
           resubmit.emit(meta.resubmit_data);
       }
{% endblock %}





