#include <stdio.h>
#include <stdlib.h>


int main(){

#ifdef _WIN32 
    printf("Please manually download the graphs and untar it from SNAP Stanford datasets\n");
    exit(0); 
#endif
    
    system("mkdir -p graphs");
    system("wget --no-check-certificate https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz -P graphs");
    system("wget --no-check-certificate https://snap.stanford.edu/data/email-Enron.txt.gz -P graphs");
    system("wget --no-check-certificate https://snap.stanford.edu/data/email-EuAll.txt.gz -P graphs");
    system("wget --no-check-certificate https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz -P graphs");
    system("wget --no-check-certificate https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz -P graphs"); 
    system("wget --no-check-certificate https://snap.stanford.edu/data/amazon0302.txt.gz  -P graphs");
    system("wget --no-check-certificate https://snap.stanford.edu/data/amazon0601.txt.gz  -P graphs");
	system("wget --no-check-certificate https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz -P graphs");
//    system("wget --no-check-certificate ");
    system("gzip -d graphs/*.gz");
//    system("sed -i '/^#/d' graphs/*");

    return 0;

}
