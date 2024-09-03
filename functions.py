def BER_calc(Rx,Tx,const):
  Rx = Rx.reshape(len(Rx),1)
  Tx = Tx.reshape(len(Tx),1)
  x1 = (const[0]+const[1])/2
  x2 = (const[1]+const[2])/2
  x3 = (const[2]+const[3])/2

  ipHat = np.zeros((len(Rx),1))
  for i in range(len(Rx)):
    if Rx[i] < x1:
      ipHat[i] = const[0]
    elif Rx[i] >= x3:
      ipHat[i] = const[3]
    elif Rx[i] >= x1 and Rx[i] < x2:
      ipHat[i] = const[1]
    elif Rx[i] >= x2 and Rx[i]<x3:
      ipHat[i] = const[2]


  pred_seq = np.zeros((len(ipHat),2))
  for i in range(len(ipHat)):
    if ipHat[i] == const[0]:
      pred_seq[i]=[0,0]
    if ipHat[i] == const[1]:
      pred_seq[i]=[0,1]
    if ipHat[i] == const[2]:
      pred_seq[i]=[1,1]
    if ipHat[i] == const[3]:
      pred_seq[i]=[1,0]

  y_test_seq = np.zeros((len(Tx),2))
  for i in range(len(Tx)):
    if Tx[i] == const[0]:
      y_test_seq[i]=[0,0]
    if Tx[i] == const[1]:
      y_test_seq[i]=[0,1]
    if Tx[i] == const[2]:
      y_test_seq[i]=[1,1]
    if Tx[i] == const[3]:
      y_test_seq[i]=[1,0]

  BER = np.sum(pred_seq.flatten() != y_test_seq.flatten())/int(len(pred_seq)*2)
  return BER

def create_dataset_symbols_multi(symbols_in, symbols_out, input_RX, output_TX, test_percent): #steps forward and backward
  if symbols_out > symbols_in:
    raise Exception("Number of output symbols is higher than the input memory!")

  # Define delay
  delay = symbols_in - symbols_out
  delay_side = int(delay/2)

  # Symbols out
  raw_size_symb_out = int((len(output_TX) - delay)//symbols_out)
  out_TX = output_TX[delay_side : delay_side + symbols_out * raw_size_symb_out]
  out_TX = out_TX.reshape((raw_size_symb_out, symbols_out))

  # Symbols in
  in_RX = input_RX[:int(raw_size_symb_out * symbols_out + delay)]
  in_RX = in_RX.reshape(len(in_RX))
  f_in = np.empty([raw_size_symb_out, symbols_in], dtype='float64')
  f_in[:] = np.nan

  for i in range(raw_size_symb_out):
    f_in[i,:] = in_RX[i * symbols_out : i * symbols_out + symbols_in]

  # Train / Test data
  test_len = int(raw_size_symb_out*(test_percent/100))

  y_train = out_TX[test_len:]
  y_test = out_TX[:test_len]

  RX_train = f_in[test_len:]
  RX_test = f_in[:test_len]

  print('Shape of Train Input_RX', RX_train.shape)
  print('Shape of Train Output_TX', y_train.shape)
  print('Shape of Train Input_RX', RX_test.shape)
  print('Shape of Train Output_TX', y_test.shape)

  return RX_train, RX_test, y_train, y_test