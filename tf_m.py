import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np;
import random
import csv
import matplotlib.pyplot as plt
import sys

########################################################################
# set up
########################################################################


span=310;
l_in=np.zeros(span);
l_pr=np.zeros(span);
m_age=np.zeros(span);
n_epochs=1;
plot_flag=1;
v_flag=2;
B_size=1;
N=1.0;
M=20;
ds_flag=0;
lr=1.e-3;# learning rate
test_frac=0.0;



########################################################################
# get options
########################################################################

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]


for i in range (0,len(opts)):
        if opts[i]=="-n_epochs": n_epochs=int(args[i]);
        if opts[i]=="-N": N=float(args[i]);
        if opts[i]=="-test_frac": test_frac=float(args[i]);
        if opts[i]=="-M": M=float(args[i]);
        if opts[i]=="-plot_flag": plot_flag=int(args[i]);
        if opts[i]=="-v_flag": v_flag=int(args[i]);
        if opts[i]=="-B_size": B_size=int(args[i]);
        if opts[i]=="-lr": lr=float(args[i]);
        if opts[i]=="-fname": fname=str(args[i]);

print("n_epochs", n_epochs);
print("test_fraction", test_frac);
print("exponent range factor N",N);
print("learning rate",lr);


########################################################################
# grab data
########################################################################


print("data source file",fname);
DataReader = csv.reader(open(fname, newline=''))  

i=0;
for row in DataReader:
	m_age[i] = float(row[1]);
	l_in[i] = float(row[2]);
	i+=1;
span=i;
print("span", span);
train_span=int(span*(1.0-test_frac))
print("train_span", train_span);
m_age-=m_age[0];
m_age/=M;


x_t = tf.convert_to_tensor(m_age, dtype=tf.float32)
y_t = tf.convert_to_tensor(l_in, dtype=tf.float32)


class UnitInterval(tf.keras.constraints.Constraint):
	def __call__(self, w):
		return tf.clip_by_value( tf.where(tf.math.is_nan(w),tf.cast(0.5,w.dtype),w) ,0.0,1.0)




class LongTail(tf.keras.layers.Layer):
	def __init__(self, num_outputs=1):
		super(LongTail, self).__init__()
		self.num_outputs = num_outputs

	def build(self, input_shape):
		self.w = self.add_weight(shape=(4, self.num_outputs),
		initializer = tf.keras.initializers.RandomUniform(minval=0.1, maxval=1.0),
		constraint=UnitInterval(),
		trainable=True);
		
	def call(self, inputs):
		a = self.w[0]/(1-self.w[1]*inputs[0]);
		a = tf.math.sign(a)*(tf.math.abs(a)**(N*self.w[2]));
		a=1-a;
		return (a);


model_1 = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(),
	LongTail(1)
])



model_1.build();
#model_1.summary();
model_1.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mean_squared_error');
model_1.fit(x_t[0:train_span],y_t[0:train_span], epochs=n_epochs,verbose=v_flag,batch_size=B_size);
w= model_1.layers[0].get_weights();
w_flag=tf.where(tf.math.is_nan(w),1,0);
nan_flag=tf.math.reduce_sum(w_flag).numpy();

if nan_flag: 
	print("hit NaN, exiting");
else:

	train_loss=model_1.evaluate(x_t[0:train_span],y_t[0:train_span], verbose=v_flag,batch_size=B_size);
	if test_frac>0:
		test_loss=model_1.evaluate(x_t[train_span:span],y_t[train_span:span], verbose=v_flag,batch_size=B_size);
	else:
		test_loss=0.0;

	print("####", w[0][0][0],-w[0][1][0]/M,N*w[0][2][0], train_loss,test_loss);

	if plot_flag:
		for i in range(0,span): l_pr[i]=model_1(m_age[i:i+1],training=False);
		plt.plot(l_in[0:span], label='actual')
		plt.plot(l_pr[0:span], label='model')
		plt.ylim([0.4,0.6]);
		plt.xlabel('time')
		plt.ylabel('labor share')
		plt.legend()
		plt.grid(True)
		plt.show();


