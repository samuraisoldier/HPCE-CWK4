#include "layer.hpp"
#include "tbb/parallel_for.h"

class ParForClusteredLayer
    : public Layer
{
protected:
    unsigned m_nIn;
    unsigned m_nOut;

    std::vector<synapse_t> m_synapses;
public:
	struct weight_and_source{ 
		//need a data type to share same weight and source for each calculation
		int32_t mass;
		uint16_t srce; 
	};
	
	struct edgeting{
		std::vector<weight_and_source> synps;	
	};
	
	std::vector<edgeting> synpsedests;
	
    ParForClusteredLayer(
        unsigned nIn,
        unsigned nOut,
        const std::vector<synapse_t> &synapses
    )
        : m_nIn(nIn)
        , m_nOut(nOut)
        , m_synapses(synapses)
		, synpsedests(m_nOut)
    {
        for(unsigned h=0; h<m_synapses.size(); h++){
            weight_and_source tempy; //declare a temporary variable 
            tempy.mass = m_synapses[h].weight; 
            tempy.srce = m_synapses[h].src; 

            synpsedests[ m_synapses[h].dst ].synps.push_back(tempy); 
        }		

	}
    
    const char *name() const
    { return "simple"; }
    
    virtual unsigned input_size() const
    { return m_nIn; }
    
    virtual unsigned output_size() const
    { return m_nOut; }
    
    void execute(
        const int8_t *pIn,  // Values of input neurons in -127..+127
        int8_t *pOut        // Values of output neurons in -127..+127
    ) const
    {        
        //std::vector<int32_t> acc(m_nOut, 0); // Create a working vector
        
        //for(unsigned i=0; i<m_synapses.size(); i++){
            // weight has 16 fractional bits, input has 7 fractional bits
            // contrib has 23 fractional bits
         //   int32_t contrib = m_synapses[i].weight * pIn[ m_synapses[i].src];
            
            // Add into acc array with 16 fractional bits
          //  acc[ m_synapses[i].dst ] += contrib >> (23-16);
			
			
        //}
        
        //for(unsigned i=0; i<m_nOut; i++){
            // Input to sigmoid has 16 fractional bits
            // Output maps to -127..+127 (7 fractional bits)
        //    pOut[i] = sigmoid( acc[i] ); // compress with sigmoid
        //}
    
	//make the outer loop
		tbb::parallel_for(0u, m_nOut, [&](unsigned i){
			int32_t acc = 0; //make a temp accumulation variable
			for(unsigned j =0; j<synpsedests[i].synps.size();j++){ 
				//calc each contrib for each weight source pair (sparse layer si most of these aren't used)
				int32_t contrib = synpsedests[i].synps[j].mass * pIn[ synpsedests[i].synps[j].srce ]; 
				//add into acc like before
				acc += contrib >> (23-16);
			}
			pOut[i] = sigmoid(acc);//sigmoid as before
		});
	}
};

LayerPtr CreateParForClusteredLayer(unsigned nIn, unsigned nOut, const std::vector<synapse_t> &synapses)
{
    return std::make_shared<ParForClusteredLayer>(nIn, nOut, synapses);
}
