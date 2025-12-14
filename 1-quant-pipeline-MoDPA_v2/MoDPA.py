#!/usr/bin/env python
# coding: utf-8
try:
    import pyopencl as cl
except:
    print("WARNING! -- no pyopencl installation, GPUs unavailable!")
import pandas as pd
import numpy as np
import os, gzip, pickle
from PTMmap import Fasta

class MoDPA:
    
    def __init__(self, name, matrix, fasta=None, thresh: tuple = (10,10), useGPU=False):
        self.thresh = thresh
        self.useGPU = useGPU
        self.name  = name
        self.fasta = fasta
        self.modpa_matrix = matrix
        self._filter_matrix()
        print("Exp Name =", self.name)
        print("MoDPA size =", self.modpa_matrix.shape)
        
        self._exps = self.modpa_matrix.columns
        self._ptms = self.modpa_matrix.index
        self.modpa_matrix = self.modpa_matrix.values
        # self._replace_UniAcc_with_Gene()
        self._randomize()

        self.modpa_corr_matrix = np.nan
        self.mock_corr_matrix = np.nan
        
        
    def calculateCorrelations(self):
        if self.useGPU:
            self.modpa_corr_matrix = MoDPA._pseudo_jaccard_on_GPU(self.modpa_matrix,self._ptms)
            self.mock_corr_matrix  = MoDPA._pseudo_jaccard_on_GPU(self.mock_matrix,self._ptms)
        else:
            self.modpa_corr_matrix = MoDPA._pseudo_jaccard(self.modpa_matrix,self._ptms) 
            self.mock_corr_matrix  = MoDPA._pseudo_jaccard(self.mock_matrix,self._ptms)
        #self.modpa_corr_matrix /= self.modpa_corr_matrix.values.max()
        #self.mock_corr_matrix  /= self.mock_corr_matrix.values.max()
        
    def save(self):
        with gzip.open(self.name+'.gz', 'wb') as outfile:
            pickle.dump(self, outfile)
            print(f">> Saved to {self.name}.gz")
    
    def load(path):
        with gzip.open(path, 'rb') as infile:
            return pickle.loads(infile.read())
    
    def getRelevantCorrelations(self, threshold=0.5):
        self.modpa_corrs = MoDPA._getCorrelations(self.modpa_corr_matrix,
                                                  threshold=threshold)
        self.mock_corrs  = MoDPA._getCorrelations(self.mock_corr_matrix,
                                                  threshold=threshold)
        self.modpa_ptms_pairs, self.mock_ptms_pairs = self._makePTMsPairLists()
        self.modpa_prot_list,  self.mock_prot_list  = self._makeProtLists()
        self.modpa_prot_pairs, self.mock_prot_pairs = self._makeProtPairLists()
    
    
    def _randomize(self):
        self.mock_matrix = self.modpa_matrix.copy()
        for row in self.mock_matrix:
            np.random.shuffle(row)
        # to shuffle columns
        self.mock_matrix = self.mock_matrix.T
        for row in self.mock_matrix:
            np.random.shuffle(row)
        self.mock_matrix = self.mock_matrix.T     
               
    def _filter_matrix(self):
        '''
        A column/ptm is kept if it contains at least self.thresh non-null, non-zero values.
        Note: Columns are filtered first!
        '''
        tmp1 = self.modpa_matrix > 0
        # col filter
        cols_to_keep = tmp1.apply(sum, axis=0) >= self.thresh[1]
        tmp2 = tmp1.loc[:,cols_to_keep]
        # row filter
        rows_to_keep = tmp2.apply(sum, axis=1) >= self.thresh[0]
        tmp3 = self.modpa_matrix.loc[rows_to_keep,cols_to_keep]

        self.modpa_matrix = tmp3

    
    def make_correlation_df(self):
        df = pd.DataFrame([_ for _ in self.modpa_corrs], 
                          columns=["ptmA","ptmB","pseudo_jaccard"])
        df.drop_duplicates(inplace=True)
        df["modA"] = df.ptmA.apply(lambda x: x.split("|")[-1])
        df["modB"] = df.ptmB.apply(lambda x: x.split("|")[-1])
        df["same_mod"] = df.modA==df.modB
        df["protA"] = df.ptmA.apply(lambda x: x.split("|")[0])
        df["protB"] = df.ptmB.apply(lambda x: x.split("|")[0])
        df["same_prot"] = df.protA==df.protB
        df.pseudo_jaccard = df.pseudo_jaccard.apply(float)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def _makeProtLists(self):
        output = []
        for corrs in [self.modpa_corrs,self.mock_corrs]:
            tmp = []
            for a,b,_ in corrs:
                tmp.append(a.split('|')[0])
                tmp.append(b.split('|')[0])
            output.append(set(tmp))
        return output[0], output[1]
           
    def _makeProtPairLists(self):
        output = []
        for corrs in [self.modpa_corrs,self.mock_corrs]:
            tmp = []
            for a,b,_ in corrs:
                a1 = a.split('|')[0]
                b1 = b.split('|')[0]
                if a==b: continue
                c = [a1,b1]
                c.sort()
                tmp.append(f"{c[0]}__{c[1]}")
            output.append(set(tmp))
        return output[0], output[1]
    
    def _makePTMsPairLists(self):
        output = []
        for corrs in [self.modpa_corrs,self.mock_corrs]:
            tmp = []
            for a,b,_ in corrs:
                if a==b: continue
                c = [a,b]
                c.sort()
                tmp.append(f"{c[0]}__{c[1]}")
            output.append(set(tmp))
        return output[0], output[1]
    
    def _replace_UniAcc_with_Gene(self):
        new_ptms = []
        for ptmid in self._ptms:
            uniacc = ptmid.split('|')[0]
            gene = Fasta.getGene(self.fasta,uniacc)
            new_ptms.append(ptmid.replace(uniacc,f"{gene}"))
        self._ptms = np.array(new_ptms)
       
    def _getCorrelations(corrMatrix, threshold):
        df = corrMatrix.fillna(0)
        corrList = []
        for ptm1,_dict in df.to_dict().items():
            for ptm2,x in _dict.items():
                if x > threshold and ptm1!=ptm2:
                    tmp = [ptm1,ptm2]
                    tmp.sort()
                    corrList.append(tuple(tmp + [f'{x:.2f}']))
        return set(corrList)

    def _pseudo_jaccard(matrix, ptms):
        print("> Calculating correlations with CPU <")
        modpa_array = matrix.T.astype(np.float32)
        pseudo_jaccard = []
        for ptm in modpa_array:
            tmp_array = modpa_array + ptm
            denominators = tmp_array.sum(axis=1)
            modpa_identity = (modpa_array>0).astype(int)
            ptm_identity   = (ptm>0).astype(int)
            numerators = np.multiply(tmp_array, modpa_identity*ptm_identity).sum(axis=1)
            pseudo_jaccard.append(np.divide(numerators,denominators))
        df = pd.DataFrame(pseudo_jaccard,
                          index=ptms, columns=ptms,
                          dtype=np.float32).round(3)
        return df.replace(1,np.nan).replace(0,np.nan)
        
    
    
    def _pseudo_jaccard_on_GPU(data, ptms):
        print("> Calculating correlations with GPU <")
        ptms_count = len(ptms)
        output = np.zeros(ptms_count**2, dtype=np.float32)

        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_CTX'] = '1'

        platform = cl.get_platforms()
        platform = [_ for _ in platform if 'Intel(R) OpenCL' in _.name]
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        matrix_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)

        prg = cl.Program(ctx, """
        __kernel void pseudojaccard(ushort nrows, ushort ncols, __global float *matrix, __global float *corr) {
            int gid = get_global_id(0);
            corr[gid] = NAN;

            int rid = gid/ncols;
            int cid = gid%ncols;

            float U = 0.0f;
            float I = 0.0f;

            if(cid!=rid) {
                for(int k=0; k<nrows; k++) {

                    float x = matrix[k*ncols + rid];
                    float y = matrix[k*ncols + cid];

                    if ((x>0) || (y>0)) {
                        U += x+y;
                    }
                    if ((x>0) && (y>0)) {
                        I += x+y;
                    }
                }
                if(I*U!=0) {
                    corr[gid] = I/U;
                } 
            }
        }
        """).build()
        prg.pseudojaccard(queue, output.shape, None,
                     np.uint32(data.shape[0]), np.uint32(data.shape[1]),
                     matrix_buf, output_buf)
        X = np.empty_like(output, dtype=np.float32)
        cl.enqueue_copy(queue, X, output_buf)
        X = X.reshape(ptms_count,ptms_count)
        
        matrix_buf.release()
        output_buf.release()
        
        return pd.DataFrame(X, index=ptms,columns=ptms, 
                            dtype=np.float32).round(3)
    
    
    def _pseudo_jaccard_on_GPU_v2(data, ptms):
        print("> Calculating correlations with GPU <")
        ptms_count = len(ptms)
        output = np.zeros(ptms_count**2, dtype=np.float32)

        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_CTX'] = '1'

        platform = cl.get_platforms()
        platform = [_ for _ in platform if 'Intel(R) OpenCL' in _.name]
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        matrix_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=output.nbytes)

        prg = cl.Program(ctx, """
        __kernel void pseudojaccard(ushort nrows, ushort ncols, __global float *matrix, __global float *corr) {
            int gid = get_global_id(0);
            corr[gid] = NAN;

            int rid = gid/ncols;
            int cid = gid%ncols;

            float U = 0.0f;
            float I = 0.0f;

            if(cid!=rid) {
                for(int k=0; k<nrows; k++) {

                    float x = matrix[k*ncols + rid];
                    float y = matrix[k*ncols + cid];

                    if ((x>1) || (y>1)) {
                        U += 1;
                    }
                    if ((x>1) && (y>1)) {
                        I += 1;
                    }
                }
                corr[gid] = I/U;
            }
        }
        """).build()
        prg.pseudojaccard(queue, output.shape, None,
                     np.uint32(data.shape[0]), np.uint32(data.shape[1]),
                     matrix_buf, output_buf)
        X = np.empty_like(output, dtype=np.float32)
        cl.enqueue_copy(queue, X, output_buf)
        X = X.reshape(ptms_count,ptms_count)
        
        matrix_buf.release()
        output_buf.release()
        
        return pd.DataFrame(X, index=ptms,columns=ptms, 
                            dtype=np.float32).round(3)