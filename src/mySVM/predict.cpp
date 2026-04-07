#include <stdlib.h>
#include <string.h>
#include <fstream.h>
#include "globals.h"
#include "example_set.h"
#include "svm_c.h"
#include "svm_nu.h"
#include "parameters.h"
#include "kernel.h"
#include "version.h"

// global svm-objects
kernel_c* kernel=0;
parameters_c* parameters=0;
svm_c* svm;
example_set_c* training_set=0;
int is_linear=1; // linear kernel?

struct example_set_list{
  example_set_c* the_set;
  example_set_list* next;
};
example_set_list* test_sets = 0;


void print_help(){
  cout<<endl;
  cout<<"predict: predict a set of examples with a trained SVM."<<endl<<endl;
  cout<<"usage: predict"<<endl
      <<"       predict <FILE>"<<endl
      <<"       predict <FILE1> <FILE2> ..."<<endl<<endl;
  cout<<"The input has to consist of:"<<endl
      <<"- the svm parameters"<<endl
      <<"- the kernel definition"<<endl
      <<"- the training result set"<<endl
      <<"- one or more sets to predict"<<endl;

  cout<<endl<<"See the documentation for the input format. The first example set to be entered is considered to be the training set, all others are test sets. Each input file can consist of one or more definitions. If no input file is specified, the input is read from <stdin>."<<endl<<endl;

  cout<<endl<<"This software is free only for non-commercial use. It must not be modified and distributed without prior permission of the author. The author is not responsible for implications from the use of this software."<<endl;
  exit(0);
};


void read_input(istream& input_stream, char* filename){
  // returns number of examples sets read
  char* s = new char[MAXCHAR];
  char next;
  next = input_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = input_stream.get(); 
  };
  while(! input_stream.eof()){
    if('#' == next){
      // ignore comment
      input_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore newline
      next = input_stream.get();
    }
    else if('@' == next){
      // new section
      input_stream >> s;
      if(0==strcmp("@parameters",s)){
	// read parameters
	if(parameters == 0){
	  parameters = new parameters_c();
	  input_stream >> *parameters;
	}
	else{
	  cout <<"*** ERROR: Parameters multiply defined"<<endl;
	  throw input_exception();
	};
      }
      else if(0==strcmp("@examples",s)){
	if(0 == training_set){
	  // input training set
	  training_set = new example_set_c();
	  if(0 != parameters){
	    training_set->set_format(parameters->default_example_format);
	  };
	  input_stream  >> *training_set;	    
	  training_set->set_filename(filename);
	  cout<<"   read "<<training_set->size()<<" examples, format "<<training_set->my_format<<", dimension = "<<training_set->get_dim()<<"."<<endl;
	}
	else{
	  // input test sets
	  example_set_list* test_set = new example_set_list;
	  test_set->the_set = new example_set_c();
	  if(0 != parameters){
	    (test_set->the_set)->set_format(parameters->default_example_format);
	  };
	  input_stream >> *(test_set->the_set);
	  (test_set->the_set)->set_filename(filename);
	  test_set->next = test_sets;
	  test_sets = test_set;
	  cout<<"   read "<<(test_set->the_set)->size()<<" examples, format "<<(test_set->the_set)->my_format<<", dimension = "<<(test_set->the_set)->get_dim()<<"."<<endl;
	};
      }
      else if(0==strcmp("@kernel",s)){
	if(0 == kernel){
	  kernel_container_c k_cont;
	  input_stream >> k_cont;
	  kernel = k_cont.get_kernel();
	}
	else{
	  cout <<"*** ERROR: Kernel multiply defined"<<endl;
	  throw input_exception();
	};
      };
    }
    else{
      // default = "@examples"
      if(0 == training_set){
	// input training set
	training_set = new example_set_c();
	if(0 != parameters){
	  training_set->set_format(parameters->default_example_format);
	};
	input_stream  >> *training_set;	    
	training_set->set_filename(filename);
	cout<<"   read "<<training_set->size()<<" examples, format "<<training_set->my_format<<", dimension = "<<training_set->get_dim()<<"."<<endl;
      }
      else{
	// input test sets
	example_set_list* test_set = new example_set_list;
	test_set->the_set = new example_set_c();
	if(0 != parameters){
	  (test_set->the_set)->set_format(parameters->default_example_format);
	};
	input_stream >> *(test_set->the_set);
	(test_set->the_set)->set_filename(filename);
	test_set->next = test_sets;
	test_sets = test_set;
	cout<<"   read "<<(test_set->the_set)->size()<<" examples, format "<<(test_set->the_set)->my_format<<", dimension = "<<(test_set->the_set)->get_dim()<<"."<<endl;
      };
    };
    next = input_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = input_stream.get(); 
    };
  };
  delete []s;
};


///////////////////////////////////////////////////////////////


int main(int argc,char* argv[]){
  cout<<"*** mySVM version "<<mysvmversion<<" ***"<<endl;
  // read objects
  try{
    if(argc<2){
      cout<<"Reading from STDIN"<<endl;
      // read vom cin
      read_input(cin,"mysvm");
    }
    else{
      char* s = argv[1];
      if((0==strcmp("-h",s)) || (0==strcmp("-help",s)) || (0==strcmp("--help",s))){
	// print out command-line help
	print_help();
      }
      else{
	// read in all input files
	for(int i=1;i<argc;i++){
	  if(0==strcmp(argv[i],"-")){
	    cout<<"Reading from STDIN"<<endl;
	    // read vom cin
	    read_input(cin,"mysvm");
	  }
	  else{
	    cout<<"Reading "<<argv[i]<<endl;
	    ifstream input_file(argv[i]);
	    if(input_file.bad()){
	      cout<<"ERROR: Could not read file \""<<argv[i]<<"\", exiting."<<endl;
	      exit(1);
	    };
	    read_input(input_file,argv[i]);
	    input_file.close();
	  };
	};
      };
    };
  }
  catch(general_exception &the_ex){
    cout<<"*** Error while reading input: "<<the_ex.error_msg<<endl;
    exit(1);
  }
  catch(...){
    cout<<"*** Program ended because of unknown error while reading input"<<endl;
    exit(1);
  };

  if(0 == parameters){
    parameters = new parameters_c();
    if(training_set->initialised_pattern_y()){
      parameters->is_pattern = 1;
      parameters->do_scale_y = 0;
    };
  };
  if(0 == kernel){
    kernel = new kernel_dot_c();
  };
  if(0 == training_set){
    cout << "*** ERROR: You did not enter the training set"<<endl;
    exit(1);
  };

  if(parameters->is_distribution){
    svm = new svm_distribution_c();
  }
  else if(parameters->is_nu){
    if(parameters->is_pattern){
      svm = new svm_nu_pattern_c();
    }
    else{
      svm = new svm_nu_regression_c();
    };
  }
  else if(parameters->is_pattern){
    svm = new svm_pattern_c();
  }
  else{
    svm = new svm_regression_c();
  };

  // scale examples
  if(parameters->do_scale){
    training_set->scale(parameters->do_scale_y);
  };

  kernel->init(parameters->kernel_cache,training_set);
  svm->init(kernel,parameters);
  svm->set_svs(training_set);

  // testing
  if(0 != test_sets){
    cout<<"----------------------------------------"<<endl;
    cout<<"Predicting"<<endl;
    example_set_c* next_test;
    SVMINT test_no = 0;
    char* outname = new char[MAXCHAR];
    while(test_sets != 0){
      test_no++;
      next_test = test_sets->the_set;
      if(training_set->initialised_scale()){
	next_test->scale(training_set->get_exp(),
			 training_set->get_var(),
			 training_set->get_dim());
      };
      if(next_test->initialised_y()){
	cout<<"Testing examples from file "<<(next_test->get_filename())<<endl;
	svm->test(next_test,1);
      };
      cout<<"Predicting examples from file "<<(next_test->get_filename())<<endl;
      svm->predict(next_test);
      // output to file .pred
      strcpy(outname,next_test->get_filename());
      strcat(outname,".pred");
      ofstream output_file(outname,
			   ios::out|ios::trunc);
      next_test->output_ys(output_file);
      output_file.close();	
      cout<<"Prediction saved in file "<<(next_test->get_filename())<<".pred"<<endl;
      test_sets = test_sets->next; // skip delete!
    };
    delete []outname;
  };

  if(parameters->verbosity > 1){
    cout << "mysvm ended successfully."<<endl;
  };
  return(0);
};
