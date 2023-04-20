#!/usr/bin/env python3

import os,sys,glob
import pandas as pd
import numpy as np
import scipy.io as sio
import nibabel as nib
from dipy.io.streamline import load_tractogram
from matplotlib import cm
import json

def load_assignments(index_file,name_file):

    return pd.merge(pd.read_csv(index_file,header=None,names=['stream_index']),pd.read_csv(name_file,header=None,names=['pair']),left_index=True,right_index=True)

def load_labels(label_file):

    return pd.read_json(label_file,orient='records')

def update_index(df):

    unique_pairs = df['pair'].unique().tolist()
    unique_indices = list(np.arange(1,len(unique_pairs)+1))
    pair_index_dict = dict(zip(unique_pairs,unique_indices))
    df['stream_index'] = df['pair'].map(pair_index_dict)

    return df, unique_pairs, unique_indices

def update_pair_names(df,labels,unique_pairs,unique_indices):
    
    track_names = []
    
    for i in unique_pairs:
        track_names.append('track_'+labels.loc[labels['label'] == int(i.split('_')[0])]['name'].values[0]+'_to_'+labels.loc[labels['label'] == int(i.split('_')[1])]['name'].values[0])
    names_index_dict = dict(zip(unique_indices,track_names))
    
    df['tract_name'] = df['stream_index'].map(names_index_dict)

    return df

def resort_pairings(df,labels):

    df = df.loc[df['pair'] != 'not-classified']
    df['pair'] = df['pair'].apply(lambda x: x.split('_')[1]+'_'+x.split('_')[0] if x != 'not-classified' and int(x.split('_')[0]) > int(x.split('_')[1]) else x)
    
    return update_pair_names(update_index(df)[0],labels,update_index(df)[1],update_index(df)[2])

def build_wmc_classification(df,track):

    bundle_names = df['tract_name'].unique().tolist()
    names = np.array(bundle_names,dtype=object)

    # generate tracts
    colors = np.reshape([np.random.random(len(bundle_names)).tolist(),np.random.random(len(bundle_names)).tolist(),np.random.random(len(bundle_names)).tolist()],(len(bundle_names),3)).tolist()
    
    streamline_index = np.zeros(len(track.streamlines))
    tractsfile = []

    for bnames in range(np.size(bundle_names)):
        tract_ind = df.loc[df['tract_name'] == bundle_names[bnames]]['stream_index'].index.tolist()
        streamline_index[tract_ind] = df.loc[df['tract_name'] == bundle_names[bnames]]['stream_index'].unique()[0]
        streamlines = np.zeros([len(track.streamlines[tract_ind])],dtype=object)
        for e in range(len(streamlines)):
            streamlines[e] = np.transpose(track.streamlines[tract_ind][e]).round(2)

        color=colors[bnames]
        count = len(streamlines)

        jsonfibers = np.reshape(streamlines[:count], [count,1]).tolist()
        for i in range(count):
            jsonfibers[i] = [jsonfibers[i][0].tolist()]

        with open ('wmc/tracts/'+str(df.loc[df['tract_name'] == bundle_names[bnames]]['stream_index'].unique()[0])+'.json', 'w') as outfile:
            jsonfile = {'name': bundle_names[bnames], 'color': color, 'coords': jsonfibers}
            json.dump(jsonfile, outfile)

        tractsfile.append({"name": bundle_names[bnames], "color": color, "filename": str(df.loc[df['tract_name'] == bundle_names[bnames]]['stream_index'].unique()[0])+'.json'})

    with open ('wmc/tracts/tracts.json', 'w') as outfile:
        json.dump(tractsfile, outfile, separators=(',', ': '), indent=4)

    # save classification structure
    sio.savemat('wmc/classification.mat', { "classification": {"names": np.reshape(names,(len(names),1)), "index": np.reshape(streamline_index,(len(streamline_index),1)) }})

def main():

    # load config.json
    with open('config.json','r') as config_f:
        config = json.load(config_f)

    # parse config.json
    label_file = config['labels'] 
    index_file = config['index']
    name_file = config['name']

    # load dwi and tractogram data
    dwi = nib.load(config['dwi'])
    tractogram = load_tractogram(config['track'],dwi,bbox_valid_check=False)

    # make output direcotry
    if not os.path.isdir('wmc'):
        os.mkdir('wmc')
        os.mkdir('wmc/tracts')
    outdir='wmc'

    # load and update assignments file to create wmc structure
    print('updating assignments for white matter classification (wmc) structure')
    assignments_df = load_assignments(index_file,name_file)
    labels = load_labels(label_file)
    final_df = resort_pairings(assignments_df,labels)

    # save wmc classificaiton
    print("building white matter classification (wmc) structure")
    build_wmc_classification(final_df,tractogram)

if __name__ == "__main__":
    main()
