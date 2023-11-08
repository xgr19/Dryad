{% extends "base.p4" %}

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
            default: parse_ethernet;
        }
    }
    
	
{% endblock %}