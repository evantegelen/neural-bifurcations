import torch
import numpy as np
import random

class Dataframe:

    def __init__(self, 
                 data_t, 
                 data_x,
                 data_a, 
                 variables=0, 
                 drivers=0,
                 batchlength=100,
                 batchsize=10):
        """
        Initialize the DataFrame with a tensors of shape (T,B,D). 
            Note this requires all timeseries to have the same time-step and length.
            If otherwise necessary rewrite of code is necessary
        """
        if not isinstance(data_x, torch.Tensor) or not isinstance(data_t, torch.Tensor):
            raise TypeError("Data must be a tensor.")
        if not len(data_x.shape) == 3:
            raise ValueError("The shape of the tensor must be 3 (T,B,D).")
        if not data_x.shape[0]%batchlength==0:
            raise ValueError("Batchlength should evenly divide time points")
        if not (data_x.shape[1]*(data_x.shape[0]/batchlength))%batchsize==0:
            raise ValueError("Batchsize should evenly divide total number of batchblocks")
        
        self.data_x                   = data_x.to(data_x.device)
        self.data_a                   = data_a.to(data_a.device)
        self.data_t                   = data_t.to(data_t.device)
        self.number_of_timeseries     = data_x.shape[1]
        self.number_of_timepoints     = data_x.shape[0]
        self.number_of_variables      = variables
        self.number_of_drivers        = drivers
        self.batch_length             = batchlength
        self.batch_size               = batchsize
        self.total_batch_blocks       = int(self.number_of_timeseries*self.number_of_timepoints/self.batch_length)
        self.batch_blocks_flattened,self.batch_blocks_a   = self.get_batch_blocks_flattened()
        self.total_batches            = self.total_batch_blocks/self.batch_size

        #Generates a random list of order of batchblocks during batch training
    def get_batch_order(self):
        orderedlist=list(range(self.total_batch_blocks))
        random.shuffle(orderedlist)
        orderedlist = [orderedlist[self.batch_size*i:self.batch_size*(i+1)] for i in range(int(self.total_batch_blocks/self.batch_size) )]
        return orderedlist
    
        #Createds flattened list of batch blocks
    def get_batch_blocks_flattened(self):
        flattened_batch_blocks=[]
        flattened_batch_a =[]
        timeseries_split=torch.chunk(self.data_x,self.number_of_timeseries,dim=1)
        counter=0
        for split in timeseries_split:
            batchlength_split = torch.chunk(split,int(self.number_of_timepoints/self.batch_length),dim=0)
            for batch_block in batchlength_split:
                flattened_batch_blocks.append(batch_block)
                flattened_batch_a.append(self.data_a[counter])
            counter+=1
        return flattened_batch_blocks,flattened_batch_a

        #Get list with batches for batchtraining
    def get_batch_list(self):
        batch_order=self.get_batch_order()
        batch_list=[]
        a_list=[]
        for batch in batch_order:
            batch_x=[]
            batch_a=[]
            for batch_block in batch:
                batch_x.append(self.batch_blocks_flattened[batch_block])
                batch_a.append(self.batch_blocks_a[batch_block])
            batch_x=torch.stack(batch_x,dim=1).squeeze()
            batch_list.append(batch_x)
            a_list.append(torch.tensor(batch_a))  
        batch_t=self.data_t[:self.batch_length]   
        return batch_list,a_list,batch_t

