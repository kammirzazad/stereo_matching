// ----------------------------------------------------------------------------------------
//    Functional Software Reference for MEMOCODE 2013 Design Contest 
//
//    This code was taken from nghiaho.com/?page_id=1366
//
//    It has been modified for MEMOCODE 2013 design contest. 
//    It reads in data cost from vdata_in.txt, and writes resulting 
//    depth labels to output_labels.txt
//
//    License from the original code is included below. 
// ----------------------------------------------------------------------------------------

/* 

   Code is released under Simplified BSD License.

   -------------------------------------------------------------------------------
   Copyright 2012 Nghia Ho. All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are
   permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other materials
   provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY NGHIA HO ``AS IS'' AND ANY EXPRESS OR IMPLIED
   WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
   FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NGHIA HO OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   The views and conclusions contained in the software and documentation are those of the
   authors and should not be interpreted as representing official policies, either expressed
   ior implied, of Nghia Ho.

*/

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <ctime>
#include <xmmintrin.h>   // Use simd extensions to speed up execution
#include <omp.h>

using namespace std;

enum DIRECTION {LEFT, RIGHT, UP, DOWN, DATA};
typedef unsigned int TYPE;
#define MAX_NUM -1 

//---------These parameters depend on picture but are fixed --------------------------//
const int S_BPS            = 36;
const int LEVELS           = 5;
const int LABELS           = 16;
const int LAMBDA           = 60;
const int BORDER_SZ        = 18;
const int SMOOTHNESS_TRUNC = 2;
#define SCALE 16
#define STEPS 4
//-----------------------------------------------------------------------------------//

struct Vertex
{
	// Each vertex has 4 messages from its
	// right/left/up/down edges and a data cost.
	TYPE msg[5][LABELS];
	int  best_assignment;
};

struct MRF2D
{
	std::vector <Vertex> grid;
	int width, height, width_tile;
};

//--- Function Prototypes --------------------------------------------------//

void InitGraph     ( const std::string &vdata_file, MRF2D  mrf[] ,int levels ); 
void WriteResults  ( const std::string &vdata_file, MRF2D &mrf   );
TYPE SmoothnessCost( int i , int j);
TYPE MAP           ( MRF2D &mrf , MRF2D &old);

void BP_joined     ( MRF2D &mrf, MRF2D &old               );
void BP_S          ( MRF2D &mrf, MRF2D &old, int factor   );

void SendMsgU      ( MRF2D &mrf, int x, int y );
void SendMsgD      ( MRF2D &mrf, int x, int y );
void SendMsgR      ( MRF2D &mrf, MRF2D &old, int x, int y );
void SendMsgL      ( MRF2D &mrf, MRF2D &old, int x, int y );

void SendMsgLR( MRF2D &mrf, MRF2D &old, int y );
void SendMsgUD( MRF2D &mrf, int x);
//--------------------------------------------------------------------------//

int main() {

	timeval t1 , t2 ;
	MRF2D mrf[LEVELS+1];
	InitGraph("vdata_in.txt" , mrf , LEVELS);

	omp_set_num_threads(12);  // Optimize code for execution on Xeon Processor

	// ------------------------------------------------------
	// FOR THE CONTEST, START IMPLEMENTATION HEREi
	// Runtime measurement starts here
	gettimeofday( &t1 , NULL);

	// Initializes dataCosts for second level using level 0's dataCosts
    #pragma omp parallel for
	for( int xy=0; xy<mrf[0].grid.size(); xy+=STEPS)
	{
		int  x = xy % mrf[0].width;
		int  y = xy / mrf[0].width;
		int index = (int)(y/4)*mrf[2].width_tile+(int)(x/4);

		for( int l=0; l<LABELS; l++)
			mrf[2].grid[index].msg[DATA][l] += 2*mrf[0].grid[xy].msg[DATA][l];
	}

	//data pyramid
	for(int i=3; i<LEVELS; i++)
	{	
        #pragma omp parallel for 
		for(int xy=0; xy<mrf[i].grid.size(); xy++)
		{
			int      x =  xy % mrf[i].width ;
			int      y =  xy / mrf[i].width ;
			int xy_old =  2*(y*mrf[i-1].width+x);

			for(int l=0; l < LABELS; l++)
			{
				// Image compression allows us to only use two pixels from four pixels
				mrf[i].grid[xy].msg[DATA][l] =  2*mrf[i-1].grid[xy_old                 ].msg[DATA][l];
				mrf[i].grid[xy].msg[DATA][l] += 2*mrf[i-1].grid[xy_old+mrf[i-1].width+1].msg[DATA][l];
			}
		}
	}

	//-------- Run bp from coarse to fine ------//            	         <>
	BP_joined ( mrf[4] , mrf[5] );              // Processing Level 4  /*    /\	*/
	BP_joined ( mrf[3] , mrf[4] );              // Processing Level 3  /*   /  \	*/
	BP_S      ( mrf[2] , mrf[3] , S_BPS );      // Processing Level 2  /*  /____\	*/
	//------------------------------------------//     		   /* ////\\\\  */ 

	gettimeofday( &t2 , NULL);
	cout << " Overall time = " <<(((t2.tv_sec -t1.tv_sec)*1000000.0) + ((t2.tv_usec - t1.tv_usec))) <<" us (micro seconds)" << endl;

	// Runtime measurement ends here 
	// FOR THE CONTEST, END IMPLEMENTATION HERE
	// ------------------------------------------------------
	// Assign labels
	TYPE energy = MAP( mrf[0] , mrf[2] );
	cout << " Energy = " << energy << endl;
	cout << " Saving results to \"" << "output_labels.txt"<<"\"."<<endl;
	WriteResults("output_labels.txt", mrf[0]);
	return 0;
}

void InitGraph(const std::string &vdata_file, MRF2D mrf[] , int levels)
{
	FILE *fp;
	int width, height;

	// Open File
	if( ( fp = fopen( vdata_file.c_str(), "r" ) ) == NULL )
	{
		printf( " Can't open file %s\n", vdata_file.c_str());
		assert(0);
	}

	fscanf(fp, "%d", &width);
	fscanf(fp, "%d", &height);

	mrf[0].width      = width;
	mrf[0].width_tile = width;
	mrf[0].height     = height;

	// Allocate grid of size width x height
	mrf[0].grid.resize(mrf[0].width * mrf[0].height);

	// Initialise edge data (messages) to zero
	for(int i=0; i < mrf[0].grid.size(); i++)
	{
		for(int j=0; j < 5; j++)
		{
			for(int k=0; k < LABELS; k++) 
				mrf[0].grid[i].msg[j][k] = 0;
		}
	}

	// Initialize vertex data (Data Cost) from given file
	for(int vid=0; vid < mrf[0].grid.size(); vid++)
	{
		for(int l=0; l < LABELS; l++) 
			fscanf(fp, "%d", &mrf[0].grid[vid].msg[DATA][l]);
	}


	// Creating the Pyramid! 
	for( int i=1; i<=levels; i++)
	{
		int old_width  = mrf[i-1].width;
		int old_height = mrf[i-1].height;

		int new_width  = (int)(old_width  / 2.0 );
		int new_height = (int)(old_height / 2.0 );

		assert( new_width  >= 1);    // Exit if new_width  == 0
		assert( new_height >= 1);    // Exit if new_height == 0

		mrf[i].width      = new_width;
		mrf[i].width_tile = new_width;
		mrf[i].height     = new_height;
		mrf[i].grid.resize(new_width*new_height);
	}

	// Initialize all edge messages in pyramid to all zero
	for( int i=2; i<=levels; i++)
	{
		int max = mrf[i].grid.size();

		for( int j=0; j<max; j++)
		{     
			for( int l=0; l<LABELS; l++)
			{
				mrf[i].grid[j].msg[RIGHT][l] = 0;
				mrf[i].grid[j].msg[LEFT ][l] = 0;
				mrf[i].grid[j].msg[UP   ][l] = 0;
				mrf[i].grid[j].msg[DOWN ][l] = 0;
			} 
		}
	}	   

	fclose(fp);
}

void BP_joined( MRF2D &mrf , MRF2D &old )
{
#pragma omp parallel for 
	for(int y=0; y<mrf.height; y++)
		SendMsgLR( mrf , old , y );

#pragma omp parallel for
	for(int x=0; x<mrf.width; x++)
		SendMsgUD( mrf , x );
}

void BP_S  ( MRF2D &mrf , MRF2D &old , int factor )
{
	int temp = mrf.height;
	mrf.height /= factor;

#pragma omp parallel for
	for( int k=0; k<factor; k++)
	{
		for( int xy=0; xy<mrf.height*mrf.width-1; xy++)
		{
			int x = xy%mrf.width;
			int y = xy/mrf.width;

			SendMsgR( mrf , old , x                  , y + k*mrf.height         );
			SendMsgL( mrf , old , (mrf.width-1) - x  , (k+1)*mrf.height - y - 1 );
		}
	}

	mrf.height = temp;
}

void SendMsgLR( MRF2D &mrf, MRF2D &old, int y )
{  
	for(int x=0; x<mrf.width-1; x++)
	{  
		SendMsgR( mrf , old , (x            ) , y );
		SendMsgL( mrf , old , (mrf.width-1-x) , y );
	}
}

void SendMsgUD( MRF2D &mrf, int x)
{
	for(int y=0; y<mrf.height-1; y++)
	{
		SendMsgU( mrf , x , (mrf.height-1-y) );
		SendMsgD( mrf , x , (y             ) ); 
	}
}

void SendMsgR( MRF2D &mrf, MRF2D &old, int x, int y )
{
	TYPE minH = MAX_NUM;
	TYPE dataCost[LABELS];

	for( int l=0; l < LABELS; l++) 
	{
		// Read data cost and LEFT edge message from current level
		dataCost[l] =  mrf.grid[y*mrf.width_tile+x].msg[DATA ][l]; 		
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[LEFT ][l];
		// Read UP and DOWN edge messages from previous level
		dataCost[l] += old.grid[(int)(y/2)*(old.width_tile)+(int)(x/2)].msg[UP   ][l];
		dataCost[l] += old.grid[(int)(y/2)*(old.width_tile)+(int)(x/2)].msg[DOWN ][l];

		minH = std::min(minH, dataCost[l]);                 

		if( l > 0 )	{ dataCost[l] = std::min(dataCost[l],dataCost[l-1]+LAMBDA); }
	}

	for( int l=LABELS-2; l >= 0; l--)
	{
		dataCost[l] = std::min(dataCost[l],dataCost[l+1]+LAMBDA);
		mrf.grid[y*mrf.width_tile + x+1].msg[LEFT][l] = std::min( dataCost[l] , minH + LAMBDA*SMOOTHNESS_TRUNC );
	}

	mrf.grid[y*mrf.width_tile + x+1].msg[LEFT][LABELS-1] = std::min( dataCost[LABELS-1] , minH + LAMBDA*SMOOTHNESS_TRUNC ); 
} 

void SendMsgL( MRF2D &mrf, MRF2D &old, int x, int y )
{
	TYPE minH = MAX_NUM;
	TYPE dataCost[LABELS];

	for( int l=0; l < LABELS; l++) // forward pass + dataCost  initialization
	{
		// Read data cost and LEFT edge message from current level
		dataCost[l] =  mrf.grid[y*mrf.width_tile+x].msg[DATA ][l]; // Data Cost
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[RIGHT][l];
		// Read UP and DOWN edge messages from previous level	
		dataCost[l] += old.grid[(int)(y/2)*(old.width_tile)+(int)(x/2)].msg[UP   ][l];
		dataCost[l] += old.grid[(int)(y/2)*(old.width_tile)+(int)(x/2)].msg[DOWN ][l];

		minH = std::min(minH, dataCost[l]); 

		if( l > 0 ) { dataCost[l] = std::min(dataCost[l],dataCost[l-1]+LAMBDA); }
	}

	// backward pass
	for( int l=LABELS-2; l >= 0; l--)
	{
		dataCost[l] = std::min(dataCost[l],dataCost[l+1]+LAMBDA);
		mrf.grid[y*mrf.width_tile + x-1].msg[RIGHT][l] = std::min( dataCost[l] , minH + LAMBDA*SMOOTHNESS_TRUNC );
	}

	//backward pass for last element
	mrf.grid[y*mrf.width_tile + x-1].msg[RIGHT][LABELS-1] = std::min(dataCost[LABELS-1],minH+ LAMBDA*SMOOTHNESS_TRUNC);
}

void SendMsgU( MRF2D &mrf, int x, int y )
{
	TYPE minH = MAX_NUM;
	TYPE dataCost[LABELS];

	for( int l=0; l < LABELS; l++) 
	{
		// Read data cost and all of edge messages from current level
		dataCost[l] =  mrf.grid[y*mrf.width_tile+x].msg[DATA ][l]; 
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[RIGHT][l];
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[LEFT ][l]; 
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[DOWN ][l];

		minH = std::min(minH, dataCost[l]); 

		if( l > 0 ) { dataCost[l] = std::min(dataCost[l],dataCost[l-1]+LAMBDA); }
	}

	for(int l=LABELS-2; l >= 0; l--)
	{
		dataCost[l] = std::min(dataCost[l],dataCost[l+1]+LAMBDA);
		mrf.grid[(y-1)*mrf.width_tile + x].msg[DOWN][l] = std::min(dataCost[l],minH+ LAMBDA*SMOOTHNESS_TRUNC);
	}

	mrf.grid[(y-1)*mrf.width_tile + x].msg[DOWN][LABELS-1] = std::min(dataCost[LABELS-1],minH+ LAMBDA*SMOOTHNESS_TRUNC);
}

void SendMsgD( MRF2D &mrf, int x, int y )
{
	TYPE minH = MAX_NUM;
	TYPE dataCost[LABELS];

	for( int l=0; l < LABELS; l++)
	{
		// Read data cost and all of edge messages from current level
		dataCost[l] =  mrf.grid[y*mrf.width_tile+x].msg[DATA ][l]; 
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[RIGHT][l];
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[LEFT ][l];
		dataCost[l] += mrf.grid[y*mrf.width_tile+x].msg[UP   ][l];

		minH = std::min(minH, dataCost[l]);  

		if( l > 0 ) { dataCost[l] = std::min(dataCost[l],dataCost[l-1]+LAMBDA); }
	}

	for( int l=LABELS-2; l >= 0; l--)
	{
		dataCost[l] = std::min(dataCost[l],dataCost[l+1]+LAMBDA);
		mrf.grid[(y+1)*mrf.width_tile + x].msg[UP][l] = std::min(dataCost[l],minH+ LAMBDA*SMOOTHNESS_TRUNC);
	}

	mrf.grid[(y+1)*mrf.width_tile + x].msg[UP][LABELS-1] = std::min(dataCost[LABELS-1],minH+ LAMBDA*SMOOTHNESS_TRUNC);
}

TYPE SmoothnessCost( int i , int j) 
{
	return LAMBDA*std::min( abs(i-j) , SMOOTHNESS_TRUNC );
}

TYPE MAP(MRF2D &mrf , MRF2D &old)
{
	// Finds the MAP assignment as well as calculating the energy
	// MAP assignment
	// Use "old" 's data cost and edge messages to calculate energy of "mrf"
	for( size_t x=0; x < mrf.width; x++)
	{ 
		for( size_t y=0; y < mrf.height; y++)
		{ 
			TYPE best = MAX_NUM;

			for( int l=0; l<LABELS; l++)
			{  
				TYPE cost=0;
				// Calculate index corresponding to (x,y) in "old"
				int  index = (int)(y/4)*(old.width)+(int)(x/4) ;

				cost += old.grid[index].msg[LEFT ][l];
				cost += old.grid[index].msg[RIGHT][l];
				cost += old.grid[index].msg[UP   ][l];
				cost += old.grid[index].msg[DOWN ][l];
				cost += old.grid[index].msg[DATA ][l];

				if( cost < best )
				{
					best = cost;
					mrf.grid[y*mrf.width+x].best_assignment = l;
				} 

			} 
		}
	}

	int width = mrf.width;
	int height = mrf.height;

	// Energy
	TYPE energy = 0;

	for( int y=0; y < mrf.height; y++)
	{
		for( int x=0; x < mrf.width; x++)
		{
			int cur_label = mrf.grid[y*width+x].best_assignment;

			// Data cost
			energy += mrf.grid[y*width+x].msg[DATA][cur_label];

			if(x-1 >= 0)     energy += SmoothnessCost(cur_label, mrf.grid[y*width+x-1  ].best_assignment);
			if(x+1 < width)  energy += SmoothnessCost(cur_label, mrf.grid[y*width+x+1  ].best_assignment);
			if(y-1 >= 0)     energy += SmoothnessCost(cur_label, mrf.grid[(y-1)*width+x].best_assignment);
			if(y+1 < height) energy += SmoothnessCost(cur_label, mrf.grid[(y+1)*width+x].best_assignment);
		}
	}

	return energy;
}

void WriteResults(const std::string &edata_file, MRF2D &mrf)
{
	FILE *fp;

	// Open File
	if( ( fp = fopen( edata_file.c_str(), "w" ) ) == NULL )
	{
		printf( " Can't open file %s\n", edata_file.c_str());
		assert(0);
	}

	// First line, write number of entries in file
	fprintf(fp, "%d\n", (mrf.width-(BORDER_SZ*2))*(mrf.height-(BORDER_SZ*2)));

	// Write label assignments
	for(int y=BORDER_SZ; y < mrf.height-BORDER_SZ; y++)
	{
		for(int x=BORDER_SZ; x < mrf.width-BORDER_SZ; x++) 
			fprintf(fp, "%d\n", mrf.grid[y*mrf.width+x].best_assignment);
	}

	fclose(fp);
}
