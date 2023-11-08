/* -*- P4_16 -*- */

#include <core.p4>
#include <tna.p4>

/*************************************************************************
 ************* C O N S T A N T S    A N D   T Y P E S  *******************
**************************************************************************/
const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8> TYPE_TCP = 6;
const bit<8> TYPE_UDP = 17;
{% block loopbackport %}{% endblock %}

/* Table Sizes */

/*************************************************************************
 ***********************  H E A D E R S  *********************************
 *************************************************************************/

/*  Define all the headers the program will recognize             */
/*  The actual sets of headers processed by each gress can differ */

/* Standard ethernet header */
header ethernet_h {
    bit<48>   dst_addr;
    bit<48>   src_addr;
    bit<16>   ether_type;
}

header ipv4_h {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  total_len;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  frag_offset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdr_checksum;
    bit<32>  src_addr;
    bit<32>  dst_addr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

{% block header_recicle_extra_info %}{% endblock %}
{% block header_resubmit_extra_info %}{% endblock %}
/*************************************************************************
 **************  I N G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/
 
    /***********************  H E A D E R S  ************************/

struct my_ingress_headers_t {
	{% block meta_recicle_extra_info %}{% endblock %}
    ethernet_h      ethernet;
    ipv4_h          ipv4;
    tcp_t           tcp;
	udp_t           udp;
}

    /******  G L O B A L   I N G R E S S   M E T A D A T A  *********/

struct my_ingress_metadata_t {
	{% block meta_resubmit_extra_info %}{% endblock %}
    bit<8>          threshold_flag;
    bit<16>          prev_node_id;
    bit<8>          class_id;
	bit<16>          srcPort;
	bit<16>          dstPort;
	bit<8>          flag;
}


    /***********************  P A R S E R  **************************/
parser IngressParser(packet_in                         pkt,
                     /* User */    
                     out my_ingress_headers_t          hdr,
                     out my_ingress_metadata_t         meta,
                     /* Intrinsic */
                     out ingress_intrinsic_metadata_t  ig_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
   
   {% block start_parse %}{% endblock %}
   
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4:  parse_ipv4;
            default: reject;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default: reject;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
		meta.srcPort=hdr.tcp.srcPort;
		meta.dstPort=hdr.tcp.dstPort;
		meta.flag=hdr.tcp.flags;
        transition accept;
    }
	
	state parse_udp {
        pkt.extract(hdr.udp);
		meta.srcPort=hdr.udp.srcPort;
		meta.dstPort=hdr.udp.dstPort;
		meta.flag=0;
        transition accept;
    }
}



control Level(inout my_ingress_headers_t hdr,
              inout my_ingress_metadata_t meta,
              inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
              (bit<16> node_size)
{
    action CheckFeature(bit<16> node_id, bit<8> less_than_feature) {
        meta.prev_node_id = node_id;                       
        meta.threshold_flag = less_than_feature; 
		//meta.flag=2;		
    }                                                           

    action SetClass(bit<16> node_id, bit<8> class_id) {
         meta.class_id = class_id;
         meta.prev_node_id = node_id;
         ig_tm_md.ucast_egress_port = (bit<9>)class_id; // for debug
		  {% block extra_set_content %}{% endblock %}
         //hdr.extra_info.setInvalid();
		 //meta.flag=1;
         exit;
     }
     
    table node {
          key = {
            meta.prev_node_id: exact;
            meta.threshold_flag: exact;
            // features
            // hdr.ipv4.ihl:ternary;
            // hdr.ipv4.diffserv:ternary;
            hdr.ipv4.total_len:{{match_name[0]}};
            hdr.ipv4.protocol:{{match_name[1]}};
            // 1 bit ternary?
            // hdr.ipv4.flags[0:0]:ternary; //Preserve
            hdr.ipv4.flags[1:1]:{{match_name[2]}};  //DF
            // hdr.ipv4.flags[2:2]:{{match_name[3]}};  //MF
            // hdr.ipv4.ihl:{{match_name[4]}};
            hdr.ipv4.ttl:{{match_name[3]}};
            meta.srcPort:{{match_name[4]}};
            meta.dstPort:{{match_name[5]}};
            meta.flag[2:2]:{{match_name[6]}};  // RST
            meta.flag[1:1]:{{match_name[7]}};  // SYN
            // meta.flag[0:0]:{{match_name[12]}};  // FIN
         }
         actions = {CheckFeature; SetClass;}
         size = node_size;
     }
       
    apply {node.apply();}
}

    /***************** M A T C H - A C T I O N  *********************/

control Ingress(
    /* User */
    inout my_ingress_headers_t                       hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{

   //Level(1+2+8+30+84+198) level0; 
  {% for level in levels %}
		Level({{table_num}}){{level}};
  {% endfor %}
  
  {% block content %}{% endblock %}
    apply {
		{% for level in levels %}
		{{level}}.apply(hdr,meta,ig_tm_md);
		{% endfor %}
		{% block content2 %}{% endblock %}
		// ig_tm_md.bypass_egress = 1;
    }
}

    /*********************  D E P A R S E R  ************************/

control IngressDeparser(packet_out pkt,
    /* User */
    inout my_ingress_headers_t                       hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
	{% block resubmit_content %}{% endblock %}
    apply {
		{% block content3 %}{% endblock %}
        pkt.emit(hdr);
    }
}


/*************************************************************************
 ****************  E G R E S S   P R O C E S S I N G   *******************
 *************************************************************************/

    /***********************  H E A D E R S  ************************/

struct my_egress_headers_t {
}

    /********  G L O B A L   E G R E S S   M E T A D A T A  *********/

struct my_egress_metadata_t {
}

    /***********************  P A R S E R  **************************/

parser EgressParser(packet_in        pkt,
    /* User */
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

    /***************** M A T C H - A C T I O N  *********************/

control Egress(
    /* User */
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    apply {
    }
}

    /*********************  D E P A R S E R  ************************/

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}


/************ F I N A L   P A C K A G E ******************************/
Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;
