
## Training

### Observer Agent Training
`python train.py observer data/observer.jsonl adapters/observer`
trainable params: 12,156,928 || all params: 3,224,906,752 || trainable%: 0.3770

Dataset size: 329 examples
{'loss': 3.6551, 'grad_norm': 2.4889016151428223, 'learning_rate': 0.0, 'epoch': 0.02}
{'loss': 3.3043, 'grad_norm': 3.1699955463409424, 'learning_rate': 3.8e-05, 'epoch': 0.48}                                                                                                                      
{'loss': 1.3851, 'grad_norm': 2.157677173614502, 'learning_rate': 7.800000000000001e-05, 'epoch': 0.97}                                                                                                         
{'loss': 0.5746, 'grad_norm': 0.9508867263793945, 'learning_rate': 0.000118, 'epoch': 1.44}                                                                                                                     
{'loss': 0.4691, 'grad_norm': 0.8834539651870728, 'learning_rate': 0.00015800000000000002, 'epoch': 1.92}                                                                                                       
{'loss': 0.4262, 'grad_norm': 2.5510025024414062, 'learning_rate': 0.00019800000000000002, 'epoch': 2.39}                                                                                                       
{'loss': 0.419, 'grad_norm': 0.711534321308136, 'learning_rate': 0.00016388468056519612, 'epoch': 2.87}                                                                                                         
{'loss': 0.4002, 'grad_norm': 0.3669200539588928, 'learning_rate': 7.710494500498662e-05, 'epoch': 3.34}                                                                                                        
{'loss': 0.3951, 'grad_norm': 0.3837604522705078, 'learning_rate': 8.520613151197898e-06, 'epoch': 3.82}                                                                                                        
{'train_runtime': 192.2689, 'train_samples_per_second': 6.845, 'train_steps_per_second': 0.874, 'train_loss': 0.8980946044127146, 'epoch': 4.0}


### Responder Agent Training
`python train.py responder data/responder.jsonl adapters/responder`

trainable params: 12,156,928 || all params: 3,224,906,752 || trainable%: 0.3770

Training agent: responder
Dataset size: 128 examples
{'loss': 3.7634, 'grad_norm': 2.4234280586242676, 'learning_rate': 0.0, 'epoch': 0.06}
{'loss': 3.4775, 'grad_norm': 2.9764461517333984, 'learning_rate': 3.8e-05, 'epoch': 1.25}                                                                                                                                                           
{'loss': 1.4453, 'grad_norm': 1.8029800653457642, 'learning_rate': 7.800000000000001e-05, 'epoch': 2.5}                                                                                                                                              
{'loss': 0.5581, 'grad_norm': 1.144328236579895, 'learning_rate': 0.000118, 'epoch': 3.75}                                                                                                                                                           
{'train_runtime': 77.0035, 'train_samples_per_second': 6.649, 'train_steps_per_second': 0.831, 'train_loss': 1.7463553119450808, 'epoch': 4.0} 

### Consultant Agent Training
`python train.py consultant data/consultant.jsonl adapters/consultant`
trainable params: 12,156,928 || all params: 3,224,906,752 || trainable%: 0.3770

Training agent: consultant
Dataset size: 43 examples
{'loss': 3.9188, 'grad_norm': 2.314115047454834, 'learning_rate': 0.0, 'epoch': 0.18}
{'loss': 3.6211, 'grad_norm': 2.718939781188965, 'learning_rate': 3.8e-05, 'epoch': 3.36}                                                                                                                                                            
{'loss': 1.5513, 'grad_norm': 2.086451292037964, 'learning_rate': 7.800000000000001e-05, 'epoch': 6.73}                                                                                                                                              
{'loss': 0.6653, 'grad_norm': 1.9155826568603516, 'learning_rate': 0.000118, 'epoch': 10.0}                                                                                                                                                          
{'train_runtime': 71.0644, 'train_samples_per_second': 6.051, 'train_steps_per_second': 0.844, 'train_loss': 1.9508689562479655, 'epoch': 10.0}