{% extends 'base.p4' %}

{% block loopbackport %}
const bit<9> LOOK_BACK_PORT = 68;
{% endblock %}

{% block header_recicle_extra_info %}
	header extra_h {
    // recirculate info
    bit<8>          threshold_flag;
    bit<16>          prev_node_id;
	}
{% endblock %}


{% block meta_recicle_extra_info %}
	extra_h         extra_info;
{% endblock %}

{% block start_parse %}
state start {
        pkt.extract(ig_intr_md);
        // advance: move forward the pointer
        pkt.advance(PORT_METADATA_SIZE);
        meta = {0, 0, 0,0,0,0};
        transition port_ayalse;
    }
    
    state port_ayalse {
        transition select(ig_intr_md.ingress_port) {
			LOOK_BACK_PORT: parse_extra_info;
            default: parse_ethernet;
        }
    }
	
	state parse_extra_info
    {
        pkt.extract(hdr.extra_info);
        meta.prev_node_id = hdr.extra_info.prev_node_id;
        meta.threshold_flag = hdr.extra_info.threshold_flag;
        transition parse_ethernet;
    }
{% endblock %}

{% block extra_set_content %}
	hdr.extra_info.setInvalid();
{% endblock %}
{% block content %}
    action action_packet_add_info() {
        hdr.extra_info.prev_node_id = meta.prev_node_id;
        hdr.extra_info.threshold_flag = meta.threshold_flag;
        ig_tm_md.ucast_egress_port = LOOK_BACK_PORT;
        hdr.extra_info.setValid();
    }    
{% endblock %}

{% block content2 %}
	action_packet_add_info();
{% endblock %}