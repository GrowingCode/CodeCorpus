/*****************************************************************************
 *                        Shapeways, Inc Copyright (c) 2012
 *                               Java Source
 *
 * This source is licensed under the GNU LGPL v2.1
 * Please read http://www.gnu.org/copyleft/lgpl.html for more information
 *
 * This software comes with the standard NO WARRANTY disclaimer for any
 * purpose. Use it at your own risk. If there's a problem you get to fix it.
 *
 ****************************************************************************/

package abfab3d.io.output;

import java.util.Arrays;
import javax.vecmath.Vector3d;

import abfab3d.core.AttributeGrid;
import abfab3d.core.Grid;
import abfab3d.grid.DensityMaker;
import abfab3d.grid.DensityMakerSubvoxel;

import abfab3d.core.TriangleCollector;

//import static java.lang.Math.abs;
import static java.lang.Math.sqrt;
import static abfab3d.core.Output.printf;


/**
   generates isosurface from data defined as a function

   @author Vladimir Bulatov

   used fragments of code by Paul Bourke

*/
public class IsosurfaceMaker {

    static final int edgeTable[] = IsosurfaceTables.edgeTable;
    static final int triTable[][] = IsosurfaceTables.triTable;
    // value to offset from 0.
    static final double TINY_VALUE = 1.e-4;


    public static final int CUBES = 0; 
    public static final int CUBES_V2 = 2; 
    public static final int TETRAHEDRA = 1; 
    
    public static final int INTERPOLATION_LINEAR = 0;
    public static final int INTERPOLATION_INDICATOR_FUNCTION = 1;

    int m_algorithm = CUBES;
    protected int m_interpolationAlgorithm = INTERPOLATION_LINEAR;

    AttributeGrid m_grid;

    
    protected double m_isoValue = 0.; // value of isosurface 
    protected double m_bounds[] = new double[]{-1, 1, -1, 1, -1, 1}; // bounds of the area 
    int m_nx=10, m_ny=10, m_nz=10;

    // working array 
    private Vector3d m_triangles[] = new Vector3d[15]; // max number of triagles is 5  
    private Cell m_cell = new Cell();
    // memory for 2 sequential slices (to be reused) 
    protected SliceData m_slice0, m_slice1;


    public IsosurfaceMaker(){
        
        for(int i = 0; i < m_triangles.length; i++){
            // this is needed for tetrahedra only 
            m_triangles[i] = new Vector3d();
        }
    }

    /**
       set bounds of area where isosurface is made
     */
    public void setBounds(double bounds[]){

        m_bounds = bounds.clone();

    }

    /**
       set grid size to calculate isosurface on 
       the area is divided into [(nx-1) x (ny-1) x (nz-1)] cubes 
       data is calculated in the corners of the cubes 
     */
    public void setGridSize(int nx, int ny, int nz ){

        m_nx = nx;
        m_ny = ny;
        m_nz = nz;

    }
    
    /**
       set value of isosurface
     */
    public void setIsovalue(double isoValue){
        m_isoValue = isoValue; 
    }

    /**

       set algorithm used for isosurface extraction 
       
       possible values: CUBES, TETRAHDRA
       
     */
    public void setAlgorithm(int algorithm){
        m_algorithm = algorithm;
    }

    /**
       set interpolation algorith to use 
       INTERPOLATION_LINEAR
       or 
       INTERPOLATION_INDICATOR_FUNCTION       
     */
    public void setInterpolationAlgorithm(int algorithm){

        m_interpolationAlgorithm = algorithm;
    }

    /**
       generates isosurface from given @scalculator and passes triangles to @tcollector
       
       slice calculator is called sequentially from zmin to zmax and is expected to fill 
       the slice data 

       triangles are passed to triangle collector 
       
     */
    public void makeIsosurface(SliceCalculator scalculator, TriangleCollector tcollector){

        double 
            xmin = m_bounds[0],
            xmax = m_bounds[1],
            ymin = m_bounds[2],
            ymax = m_bounds[3],
            zmin = m_bounds[4],
            zmax = m_bounds[5];
        final int nx = m_nx;
        final int nx1 = nx-1;
        final int ny = m_ny;
        final int ny1 = ny-1;
        final int nz = m_nz;
        final int nz1 = nz-1;

        final double dx = (xmax - xmin)/nx1;
        final double dy = (ymax - ymin)/ny1;
        final double dz = (zmax - zmin)/nz1;
        
        if(true){
            //if(m_slice0 == null) {
            m_slice0 = new SliceData(m_nx, m_ny, xmin, xmax, ymin, ymax);
            m_slice1 = new SliceData(m_nx, m_ny, xmin, xmax, ymin, ymax);            
        } else {
            m_slice0.setParams(m_nx, m_ny, xmin, xmax, ymin, ymax);
            m_slice1.setParams(m_nx, m_ny, xmin, xmax, ymin, ymax);        
        }
        
        SliceData slice0 = m_slice0;
        SliceData slice1 = m_slice1;

        Cell cell = m_cell;
        Vector3d cpnt[] = cell.p; // coordinates of corners of the cube 
        double cval[] = cell.val; // values in the corners 
        Vector3d cedges[] = cell.e; // 
        Vector3d triangles[] = m_triangles;

        final double isoValue = m_isoValue;
        final int algorithm = m_algorithm;

        slice0.setZ(zmin);
        scalculator.getSlice(slice0); 

        for(int iz = 0; iz < nz1; iz++) {
            double z = zmin + dz * iz;
            double z1 = z+dz;

            slice1.setZ(z1);
            scalculator.getSlice(slice1); 
            double data0[] = slice0.data;
            double data1[] = slice1.data;

            for(int iy = 0; iy < ny1; iy++) {

                double y = ymin + dy*iy;
                int iy1 = iy+1;
                double y1 = y+dy;
                
                for(int ix = 0; ix < nx1; ix++) {

                    int ix1 = ix+1;
                    double x = xmin + dx*ix;
                    double x1 = x+dx;
                    
                    cpnt[0].set(x, y, z );
                    cpnt[1].set(x1,y, z );
                    cpnt[2].set(x1,y, z1);
                    cpnt[3].set(x, y, z1);
                    cpnt[4].set(x, y1,z );
                    cpnt[5].set(x1,y1,z );
                    cpnt[6].set(x1,y1,z1);
                    cpnt[7].set(x, y1,z1);

                    int base = ix  + iy * nx;  // offset of point (x,y)
                    int base1 = base + nx;     // offset of point (x, y+1)

                    cval[0] = data0[base] - isoValue;
                    cval[1] = data0[base + 1] - isoValue;
                    cval[2] = data1[base + 1] - isoValue;
                    cval[3] = data1[base] - isoValue;
                    cval[4] = data0[base1] - isoValue;
                    cval[5] = data0[base1 + 1] - isoValue;
                    cval[6] = data1[base1 + 1] - isoValue;
                    cval[7] = data1[base1] - isoValue;

                    shiftFromZero(cval);
                    
                    switch(algorithm){
                    default:
                    case CUBES:
                        polygonizeCube(cell, triangles, tcollector);
                        break;
                    case CUBES_V2:
                        polygonizeCube_v2(cval, cpnt, cedges, triangles, tcollector);
                        break;
                    case TETRAHEDRA:
                        polygonizeCube_tetra(cell, 0., triangles, tcollector);
                        break;
                    }                    
                }// ix
            } // iy 

            // switch calculated slices
            SliceData stmp = slice0;
            slice0 = slice1;
            slice1 = stmp;            

        }  // for(iz...           
    }

    
    static final double ISOEPS = 1.e-2;

    static final double abs(double v) {
        return ( v >=0.)? v: -v;
    }
        
    /**
       shifts values close to zero to ISOEPS distance above zero 
       this is to prevent triangles to have vertivces exactly in the cube corners 
     */
    public static void shiftFromZero(double v[]){

        for(int i = 0; i < v.length; i++){
            if(abs(v[i]) < ISOEPS){
                v[i] = ISOEPS; 
            }
        }        
    }
    
    //
    // this version produces no garbage 
    //
    public void polygonizeCube(Cell g, Vector3d triangles[], TriangleCollector ggen){

        int cubeindex = 0;
        double iso = 0.0;

        if (g.val[0] < 0) cubeindex |= 1;
        if (g.val[1] < 0) cubeindex |= 2;
        if (g.val[2] < 0) cubeindex |= 4;
        if (g.val[3] < 0) cubeindex |= 8;
        if (g.val[4] < 0) cubeindex |= 16;
        if (g.val[5] < 0) cubeindex |= 32;
        if (g.val[6] < 0) cubeindex |= 64;
        if (g.val[7] < 0) cubeindex |= 128;

        /* Cube is entirely in/out of the surface */
        if (edgeTable[cubeindex] == 0)
            return;

        /* Find the vertices where the surface intersects the cube */
        if ((edgeTable[cubeindex] & 1)  != 0)
            vertexInterp(iso,g.p[0],g.p[1],g.val[0],g.val[1],g.e[0]);

        if ((edgeTable[cubeindex] & 2) != 0)
            vertexInterp(iso,g.p[1],g.p[2],g.val[1],g.val[2],g.e[1]);

        if ((edgeTable[cubeindex] & 4) != 0)
            vertexInterp(iso,g.p[2],g.p[3],g.val[2],g.val[3],g.e[2]);

        if ((edgeTable[cubeindex] & 8) != 0)
            vertexInterp(iso,g.p[3],g.p[0],g.val[3],g.val[0],g.e[3]);

        if ((edgeTable[cubeindex] & 16) != 0)
            vertexInterp(iso,g.p[4],g.p[5],g.val[4],g.val[5],g.e[4]);

        if ((edgeTable[cubeindex] & 32) != 0)
            vertexInterp(iso,g.p[5],g.p[6],g.val[5],g.val[6],g.e[5]);

        if ((edgeTable[cubeindex] & 64) != 0)
            vertexInterp(iso,g.p[6],g.p[7],g.val[6],g.val[7],g.e[6]);

        if ((edgeTable[cubeindex] & 128) != 0)
            vertexInterp(iso,g.p[7],g.p[4],g.val[7],g.val[4],g.e[7]);

        if ((edgeTable[cubeindex] & 256) != 0)
            vertexInterp(iso,g.p[0],g.p[4],g.val[0],g.val[4],g.e[8]);

        if ((edgeTable[cubeindex] & 512) != 0)
            vertexInterp(iso,g.p[1],g.p[5],g.val[1],g.val[5],g.e[9]);

        if ((edgeTable[cubeindex] & 1024) != 0)
            vertexInterp(iso,g.p[2],g.p[6],g.val[2],g.val[6],g.e[10]);

        if ((edgeTable[cubeindex] & 2048) != 0)
            vertexInterp(iso,g.p[3],g.p[7],g.val[3],g.val[7],g.e[11]);

        /* Create the triangles */
        int ntriang = 0;
        for (int i=0; i < triTable[cubeindex].length; i+=3) {

            triangles[i]   = g.e[triTable[cubeindex][i  ]];
            triangles[i+1] = g.e[triTable[cubeindex][i+1]];
            triangles[i+2] = g.e[triTable[cubeindex][i+2]];
            ntriang++;
        }

        addTri(ggen, triangles, ntriang);

    }

    protected final void polygonizeCube_v2(double val[], Vector3d p[], Vector3d e[], Vector3d triangles[], TriangleCollector ggen){

        int cubeindex = 0;
        double iso = 0.0;

        if (val[0] < 0) cubeindex |= 1;
        if (val[1] < 0) cubeindex |= 2;
        if (val[2] < 0) cubeindex |= 4;
        if (val[3] < 0) cubeindex |= 8;
        if (val[4] < 0) cubeindex |= 16;
        if (val[5] < 0) cubeindex |= 32;
        if (val[6] < 0) cubeindex |= 64;
        if (val[7] < 0) cubeindex |= 128;

        /* Cube is entirely in/out of the surface */
        if (edgeTable[cubeindex] == 0)
            return;

        /* Find the vertices where the surface intersects the cube */
        if ((edgeTable[cubeindex] & 1)  != 0)
            vertexInterp(p[0],p[1],val[0],val[1],e[0]);

        if ((edgeTable[cubeindex] & 2) != 0)
            vertexInterp(p[1],p[2],val[1],val[2],e[1]);

        if ((edgeTable[cubeindex] & 4) != 0)
            vertexInterp(p[2],p[3],val[2],val[3],e[2]);

        if ((edgeTable[cubeindex] & 8) != 0)
            vertexInterp(p[3],p[0],val[3],val[0],e[3]);

        if ((edgeTable[cubeindex] & 16) != 0)
            vertexInterp(p[4],p[5],val[4],val[5],e[4]);

        if ((edgeTable[cubeindex] & 32) != 0)
            vertexInterp(p[5],p[6],val[5],val[6],e[5]);

        if ((edgeTable[cubeindex] & 64) != 0)
            vertexInterp(p[6],p[7],val[6],val[7],e[6]);

        if ((edgeTable[cubeindex] & 128) != 0)
            vertexInterp(p[7],p[4],val[7],val[4],e[7]);

        if ((edgeTable[cubeindex] & 256) != 0)
            vertexInterp(p[0],p[4],val[0],val[4],e[8]);

        if ((edgeTable[cubeindex] & 512) != 0)
            vertexInterp(p[1],p[5],val[1],val[5],e[9]);

        if ((edgeTable[cubeindex] & 1024) != 0)
            vertexInterp(p[2],p[6],val[2],val[6],e[10]);

        if ((edgeTable[cubeindex] & 2048) != 0)
            vertexInterp(p[3],p[7],val[3],val[7],e[11]);

        /* Create the triangles */
        int ntriang = 0;
        for (int i=0; i < triTable[cubeindex].length; i+=3) {

            triangles[i]   = e[triTable[cubeindex][i  ]];
            triangles[i+1] = e[triTable[cubeindex][i+1]];
            triangles[i+2] = e[triTable[cubeindex][i+2]];
            ntriang++;
        }

        addTri(ggen, triangles, ntriang);

    }

    /*
      Polygonise a tetrahedron given its vertices within a cube
      This is an alternative algorithm to polygonisegrid.
      It results in a smoother surface but more triangular facets.
      
                      + 0
                     /|\
                    / | \
                   /  |  \
                  /   |   \
                 /    |    \
                /     |     \
               +-------------+ 1
              3 \     |     /
                 \    |    /
                  \   |   /
                   \  |  /
                    \ | /
                     \|/
                      + 2

     It can be used for polygonization of cube: 
   
      polygoniseTetra(grid,iso,triangles,0,2,3,7);
      polygoniseTetra(grid,iso,triangles,0,2,6,7);
      polygoniseTetra(grid,iso,triangles,0,4,6,7);
      polygoniseTetra(grid,iso,triangles,0,6,1,2);
      polygoniseTetra(grid,iso,triangles,0,6,1,4);
      polygoniseTetra(grid,iso,triangles,5,6,1,4);


      vertices and edges 

                                      
                   4+-------4----------+5
                   /|                 /|                                     
                  7 |                5 |                                      
                 /  |               /  |                                      
               7+---------6--------+6  9                                                         
                |   |8             |   |                                     
               11   |             10   |                                     
                |  0+-------0------|---+1
                |  /               |  /                                     
                | 3                | 1                                        
                |/                 |/                                        
               3+--------2---------+2
                                   
    */
    public void polygonizeCube_tetra(Cell cell, double iso, Vector3d triangles[], TriangleCollector ggen){

        int count;

        count = polygonizeTetra(cell, iso, 0,2,3,7, triangles); if(count > 0) addTri(ggen, triangles, count);
        count = polygonizeTetra(cell, iso, 0,6,2,7, triangles); if(count > 0) addTri(ggen, triangles, count);
        count = polygonizeTetra(cell, iso, 0,4,6,7, triangles); if(count > 0) addTri(ggen, triangles, count);
        count = polygonizeTetra(cell, iso, 0,6,1,2, triangles); if(count > 0) addTri(ggen, triangles, count);
        count = polygonizeTetra(cell, iso, 0,1,6,4, triangles); if(count > 0) addTri(ggen, triangles, count);
        count = polygonizeTetra(cell, iso, 5,6,1,4, triangles); if(count > 0) addTri(ggen, triangles, count);
        
    }


    /**
       return number of triangles via given tetrahedron
       
       triangles are stored in tri[]
       
       normals to triangle is pointed toward negative values of function
       this is to make positive valued area looks solid from outside 
              
    */
    public int polygonizeTetra(Cell g, double iso,int v0,int v1,int v2,int v3,Vector3d tri[]) {
        
        /*
          Determine which of the 16 cases we have given which vertices
          are above or below the isosurface
        */
        int triindex = 0;
        if (g.val[v0] < iso) triindex |= 1;
        if (g.val[v1] < iso) triindex |= 2;
        if (g.val[v2] < iso) triindex |= 4;
        if (g.val[v3] < iso) triindex |= 8;
        
        // Form the vertices of the triangles for each case      
        switch (triindex) {
        case 0x00:
        case 0x0F:
            return 0;
            
        case 0x0E: // 1110  01 03 02 
            vertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1], tri[0]);
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3], tri[1]);
            vertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2], tri[2]);
            return 1;
            
        case 0x01: //  0001 01 02 03 
            vertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1],tri[0]);
            vertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2],tri[1]);
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[2]);
            return 1;
            
        case 0x0D: // 1101 
            vertexInterp(iso,g.p[v1],g.p[v0],g.val[v1],g.val[v0],tri[0]);
            vertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2],tri[1]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[2]);
            return 1;
            
        case 0x02: // 0010   10 13 12 
            vertexInterp(iso,g.p[v1],g.p[v0],g.val[v1],g.val[v0],tri[0]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2],tri[2]);
            return 1;
            
        case 0x0C: // 1100  12 13 03, 12 03 02 
            vertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2],tri[0]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[2]);
            
            tri[3].set(tri[0]);
            tri[4].set(tri[2]);
            vertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2], tri[5]);
            return 2;
            
        case 0x03: // 0011  12 03 13, 12 02 03 
            vertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2],tri[0]);
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[2]);
            tri[3].set(tri[0]);
            vertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2],tri[4]);
            tri[5].set(tri[1]);
            return 2;
            
        case 0x0B: // 1011  2-> 013
            vertexInterp(iso,g.p[v2],g.p[v0],g.val[v2],g.val[v0],tri[0]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v2],g.p[v1],g.val[v2],g.val[v1],tri[2]);
            return 1;
            
        case 0x04: // 0100  2 -> 013
            vertexInterp(iso,g.p[v2],g.p[v0],g.val[v2],g.val[v0],tri[0]);
            vertexInterp(iso,g.p[v2],g.p[v1],g.val[v2],g.val[v1],tri[1]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[2]);
            return 1;
            
        case 0x0A: // 1010 
            vertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1],tri[0]);
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[2]);
            tri[3].set(tri[0]);
            tri[4].set(tri[2]);
            vertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2],tri[5]);
            return 2;        
            
        case 0x05: // 0101 
            vertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1],tri[0]);
            vertexInterp(iso,g.p[v1],g.p[v2],g.val[v1],g.val[v2],tri[1]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[2]);
            tri[3].set(tri[0]);
            tri[4].set(tri[2]);
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[5]);
            return 2;
            
        case 0x09: // 1001
            vertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1],tri[0]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[2]);
            tri[3].set(tri[0]);
            vertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2],tri[4]);
            tri[5].set(tri[1]);
            return 2;
            
        case 0x06: // 0110
            vertexInterp(iso,g.p[v0],g.p[v1],g.val[v0],g.val[v1],tri[0]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[2]);
            tri[3].set(tri[0]);
            tri[4].set(tri[2]);
            vertexInterp(iso,g.p[v0],g.p[v2],g.val[v0],g.val[v2],tri[5]);
            return 2;
            
        case 0x07: // 0111
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[0]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[2]);
            return 1;
            
        case 0x08: // 1000
            vertexInterp(iso,g.p[v0],g.p[v3],g.val[v0],g.val[v3],tri[0]);
            vertexInterp(iso,g.p[v2],g.p[v3],g.val[v2],g.val[v3],tri[1]);
            vertexInterp(iso,g.p[v1],g.p[v3],g.val[v1],g.val[v3],tri[2]);
            return 1;
        }
        return 0;
    } //polygoniseTetra()
    
    
    
    /**
       add one or two triangles to ggen 
     */
    static void addTri(TriangleCollector ggen, Vector3d tri[], int count){

        switch(count){
        default:
            return;
        case 5:
            ggen.addTri(tri[12],tri[13],tri[14]);
            // no break 
        case 4:
            ggen.addTri(tri[9],tri[10],tri[11]);      
            // no break 
        case 3:
            ggen.addTri(tri[6], tri[7], tri[8]);      
            // no break 
        case 2:
            ggen.addTri(tri[3], tri[4], tri[5]);      
            // no break 
        case 1:
            ggen.addTri(tri[0],tri[1],tri[2]);      
        }
    }

    static final double EPS = 1.e-12;

    /*
      Linearly interpolate the position where an isosurface cuts
      an edge between two vertices, each with their own scalar value
      generates no garbage 
    */
    protected final void vertexInterp(double isolevel,Vector3d p1,Vector3d p2, double valp1, double valp2, Vector3d dest){

        if (abs(isolevel-valp1) < EPS){
            dest.set(p1);
            return;
        }
        if (abs(isolevel-valp2) < EPS){
            dest.set(p2);
            return;
        }
        if (abs(valp1-valp2) < EPS){
            dest.set(p1);
            return;
        }

        double mu = getLerpCoeff(valp1, valp2, isolevel); 

        double x = lerp(p1.x,p2.x, mu); 
        double y = lerp(p1.y,p2.y, mu); 
        double z = lerp(p1.z,p2.z, mu); 
        
        dest.set(x,y,z);

    }

    protected final void vertexInterp(Vector3d p1,Vector3d p2, double valp1, double valp2, Vector3d dest){

        if (abs(valp1) < EPS){
            dest.set(p1);
            return;
        }
        if (abs(valp2) < EPS){
            dest.set(p2);
            return;
        }
        if (abs(valp1-valp2) < EPS){
            dest.set(p1);
            return;
        }

        double mu = getLerpCoeff(valp1, valp2); 

        double x = lerp(p1.x,p2.x, mu); 
        double y = lerp(p1.y,p2.y, mu); 
        double z = lerp(p1.z,p2.z, mu); 
        
        dest.set(x,y,z);

    }

    static final double lerp(double x1, double x2, double t){
        return x1 + t * (x2-x1);
    }


    protected final double getLerpCoeff(double v1, double v2, double isolevel){

        switch(m_interpolationAlgorithm){
        default:
        case INTERPOLATION_LINEAR:
            return (isolevel - v1) / (v2 - v1);

        case INTERPOLATION_INDICATOR_FUNCTION:
            return coeff_indicator(0.5*( 1- v1), 0.5*(1-v2));
        }
    }

    // coeff for zero isolevel 
    protected final double getLerpCoeff(double v1, double v2){

        switch(m_interpolationAlgorithm){
        default:
        case INTERPOLATION_LINEAR:
            return ( -v1) / (v2 - v1);

        case INTERPOLATION_INDICATOR_FUNCTION:
            return coeff_indicator(0.5*( 1- v1), 0.5*(1-v2));
        }
    }

    // indicator function algorithm is based on 
    // 
    // J. Manson, J. Smith, and S. Schaefer (2011) 
    // Contouring Discrete Indicator Functions
    //
    public static final double coeff_indicator(double v1, double v2){
        if (v1 < v2)
            return coeff_indicator_half(v1, v2);
        else
            return 1. - coeff_indicator_half(v2, v1);
    }
    
    public static final double coeff_indicator_half(double v1, double v2){

        int selector = 0;
	if (3*v1 >= v2) // test 1-3
            selector += 1;
	if (v1 + 2 >= 3*v2) // test 1-4
            selector += 2;
        
	switch (selector){
            
	case 3: // must be 1
            return (v1 - .5) / (v1 - v2);  
	case 0: // must be 2
            return 1.5 - v1 - v2;            
	case 2: // test 2-3
            {
                double d = v1*(v1+v2);
                double s = 2*v1+2*v2-1;
                if (4*d > s*s)	{
                    return 1. - (2*v2 - 1) / (8*v1 + 4*v2 -8*sqrt(d)); // must be 3
                } else {
                    return 1.5 - v1 - v2; // must be 2
                }
            }
            
	case 1: // test 2-4
            {
                double b1 = 1 - v2;
                double b2 = 1 - v1;
                
                double d = b1*(b1+b2);
                double s = 2*b1+2*b2-1;
                if (4*d > s*s){
                    return (2*b2 - 1) / (8*b1 + 4*b2 - 8*sqrt(d)); // must be 4
                } else {
                    return 1.5 - v1 - v2; // must be 2
                }
            }
	}
        
	return 0;
    }
    
    //
    // class describes one cubic cell    
    //
    public static class Cell {

        double val[]; // values at corners of the cube         
        Vector3d p[]; // coordinates of corners of the cube 
        Vector3d e[]; // coordinates of isosurface-edges intersections 
        
        Cell(){
            
            val = new double[8];
            p = new Vector3d[8];
            e = new Vector3d[12];
            
            for(int i = 0; i < p.length; i++){
                p[i] = new Vector3d();
            }
            for(int i = 0; i < e.length; i++){
                e[i] = new Vector3d();
            }
        }    
    }

    /**
       inteface to calculate one slice of data in x,y plane 

     */
    public static interface SliceCalculator {
        
        /**
           method shall fill data array of sliceData with values 
         */
        public void getSlice(SliceData sliceData);
    }


    /**
       interface to return data value at the given point 
     */
    public interface DataXYZ {

        public double getData(double x, double y, double z); 

    }

    

    /**
       data holder for one slice of data in xy-plane 
       
     */
    public static class SliceData {

        public int nx, ny;
        // bounds of the slice 
        public double xmin, ymin, xmax, ymax;  
        // data values in x,y points 
        public double data[];
        public double z; 
        
        SliceData(int nx, int ny, double xmin, double xmax, double ymin, double ymax){

            this.nx = nx;
            this.ny = ny;
            
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
            
            data = new double[(nx+1)*(ny+1)];
            if(false)printf("slice data alloc (%d %d) thread:%s\n", nx, ny, Thread.currentThread().getName());
            
        }

        void setParams(int nx, int ny, double xmin, double xmax, double ymin, double ymax){

            if(nx*nx > data.length) {
                if(false)printf("slice data realloc (%d %d) thread:%s\n", nx, ny, Thread.currentThread().getName());
                // reallocate data 
                // allocate a lttle bit more 
                data = new double[(nx+1)*(ny+1)];
            }
            this.nx = nx;
            this.ny = ny;
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
        }

        void setZ(double z){
            this.z = z;
        }

    } // class SliceData 


    /**
       class calculates slice of data from DataXYZ 
       
     */
    public static class SliceFunction implements SliceCalculator {
        
        DataXYZ fdata;
        
        public SliceFunction(DataXYZ fdata){

            this.fdata = fdata;

        }

        public void getSlice(SliceData sliceData){

            int nx = sliceData.nx;
            int ny = sliceData.ny;
            
            double xmin = sliceData.xmin;
            double ymin = sliceData.ymin;

            double dx = (sliceData.xmax - xmin)/(nx-1);
            double dy = (sliceData.ymax - ymin)/(ny-1);
            double z = sliceData.z;
            double data[] = sliceData.data;
            for(int iy = 0; iy < ny; iy++){
                    
                double y = ymin + iy * dy;
                
                int offset = iy*nx;
                
                for(int ix = 0; ix < nx; ix++){
                    
                    double x = xmin + ix * dx;
                    data[offset + ix] = fdata.getData(x,y,z);
                }
            }
                        
        }
    } // class SliceFunction 



    /**
       class calculates slice of data from a Grid
       
     */
    public static class SliceGrid implements SliceCalculator {
        

        Grid grid;
        double bounds[];
        int gnx, gny, gnz; // size of grid 
        double gdx, gdy, gdz; // pixel size of grid 
        double gxmin, gymin, gzmin; // origin of the grid 
        int m_smoothSteps = 0;

        public SliceGrid(Grid grid, double bounds[], int smoothSteps){

            this.grid = grid;
            this.bounds = bounds.clone();
            m_smoothSteps = smoothSteps;
            gnx = grid.getWidth();
            gny = grid.getHeight();
            gnz = grid.getDepth();
            
            gdx = (bounds[1] - bounds[0])/(gnx-1);
            gdy = (bounds[3] - bounds[2])/(gny-1);
            gdz = (bounds[5] - bounds[4])/(gnz-1);

            gxmin = bounds[0];
            gymin = bounds[2];
            gzmin = bounds[4];

        }

        static int round(double x){
            return (int)Math.floor(x + 0.5);
        }

        public void getSlice(SliceData sliceData){

            int nx = sliceData.nx;
            int ny = sliceData.ny;

            double xmin = sliceData.xmin;
            double ymin = sliceData.ymin;

            double dx = (sliceData.xmax - xmin)/(nx-1);
            double dy = (sliceData.ymax - ymin)/(ny-1);            

            double z = sliceData.z;

            int gz = round((z - gzmin)/gdz);
            
            double data[] = sliceData.data;
           
            for(int iy = 0; iy < ny; iy++){

                double y = ymin + iy*dy;
                int gy = round((y - gymin)/gdy);

                int offset = iy * nx;

                for(int ix = 0; ix < nx; ix++){

                    double x = xmin + ix*dx;

                    int gx = round((x - gxmin)/gdx);
                    data[offset + ix] = getGridData(gx,gy,gz, m_smoothSteps);
                }
            }            
        }

        /**
           return data at the grid point 
           does recursive averaging
         */
        double getGridData(int gx, int gy, int gz, int smoothSteps){

            if(smoothSteps == 0){
                if(gx <  0 || gy < 0 || gz < 0 || gx >= gnx || gy >= gny || gz >= gnz){
                    return 1;
                } else {
                    byte state = grid.getState(gx,gy,gz);
                    if(state == Grid.OUTSIDE)
                        return 1;
                    else 
                        return -1;
                }            
            } else {
                smoothSteps--;
                double sum = 0.;
                //double orig = getGridData(gx,   gy,   gz, smoothSteps);
                sum += getGridData(gx+1, gy,   gz, smoothSteps);
                sum += getGridData(gx-1, gy,   gz, smoothSteps);
                sum += getGridData(gx,   gy+1, gz, smoothSteps);
                sum += getGridData(gx,   gy-1, gz, smoothSteps);
                sum += getGridData(gx,   gy,   gz+1, smoothSteps);
                sum += getGridData(gx,   gy,   gz-1, smoothSteps);
                sum /= 6;
                
                return sum;
            }
        }
    } // class SliceGrid 



    /**
       class calculates slice of data from a Grid doing averaging over neighbors 
       
     */
    public static class SliceGrid2 implements SliceCalculator {

        
        Grid grid;
        double bounds[];
        int gnx, gny, gnz; // size of grid 
        double gdx, gdy, gdz; // pixel size of grid 
        double gxmin, gymin, gzmin; // origin of the grid 

        int m_cubeHalfSize = 0; // allowed values are 0, 1, 2, ...
        double m_bodyVoxelWeight = 1.0;

        public SliceGrid2(Grid grid, double bounds[], int resamplingFactor){
            this(grid,bounds, resamplingFactor, 1.0);
        }

        /**
           body voxels may have larger weight ( > 1. ) or smaller weight ( < 1.)  
         */
        public SliceGrid2(Grid grid, double bounds[], int resamplingFactor, double bodyVoxelWeight){

            this.grid = grid;
            this.bounds = bounds.clone();
            m_cubeHalfSize = resamplingFactor / 2;
            m_bodyVoxelWeight = bodyVoxelWeight;

            gnx = grid.getWidth();
            gny = grid.getHeight();
            gnz = grid.getDepth();
            
            gdx = (bounds[1] - bounds[0])/(gnx-1);
            gdy = (bounds[3] - bounds[2])/(gny-1);
            gdz = (bounds[5] - bounds[4])/(gnz-1);

            gxmin = bounds[0];
            gymin = bounds[2];
            gzmin = bounds[4];

        }

        static int round(double x){
            return (int)Math.floor(x + 0.5);
        }

        public void getSlice(SliceData sliceData){

            int nx = sliceData.nx;
            int ny = sliceData.ny;

            double xmin = sliceData.xmin;
            double ymin = sliceData.ymin;

            double dx = (sliceData.xmax - xmin)/(nx-1);
            double dy = (sliceData.ymax - ymin)/(ny-1);            

            double z = sliceData.z;

            int gz = round((z - gzmin)/gdz);
            
            double data[] = sliceData.data;
           
            for(int iy = 0; iy < ny; iy++){

                double y = ymin + iy*dy;
                int gy = round((y - gymin)/gdy);

                int offset = iy * nx;

                for(int ix = 0; ix < nx; ix++){

                    double x = xmin + ix*dx;

                    int gx = round((x - gxmin)/gdx);
                    data[offset + ix] = getGridData(gx,gy,gz);
                }
            }            
        }

        /**
           return data at the grid point 
           does averaging
         */
        double getGridData(int gx, int gy, int gz){

            double sum =0; 
            double norm = 0;
            for(int dx = -m_cubeHalfSize; dx <= m_cubeHalfSize; dx++){
                for(int dy = -m_cubeHalfSize; dy <= m_cubeHalfSize; dy++){
                    for(int dz = -m_cubeHalfSize; dz <= m_cubeHalfSize; dz++){
                        double v = getGridState(gx+dx, gy+dy, gz+dz);
                        if( v < 0.0){// body voxel 
                            sum += v * m_bodyVoxelWeight; 
                            norm += m_bodyVoxelWeight;
                        } else {
                            sum += v; 
                            norm += 1;
                        }
                    }
                }
            }
            if(abs(sum) < TINY_VALUE){
                sum = (sum > 0.)? TINY_VALUE: -TINY_VALUE;
                //printf("zero sum in getGridData: %g\n", sum);
            }
            return sum * norm;

        }

        int getGridState(int gx, int gy, int gz){

            if(gx <  0 || gy < 0 || gz < 0 || gx >= gnx || gy >= gny || gz >= gnz){
                return 1; // outside
            } else {
                byte state = grid.getState(gx,gy,gz);
                if(state == Grid.OUTSIDE)
                    return 1;
                else 
                    return -1;
            }            
        }
    } // class SliceGrid2


    /**
       
       allocates copy of block of grid and does optional convolution over that block using 
       given kernel 

     */
    public static class BlockSmoothingSlices implements SliceCalculator {


        AttributeGrid agrid;
        Grid grid;
        double bounds[] = new double[6];


        int gnx, gny, gnz; // size of grid 
        double gdx, gdy, gdz; // pixel size of grid 
        double gxmin, gymin, gzmin; // origin of the grid 
        
        double blockData[]; // data of the block 
        double rowData[];// data for one row for convolution 

        // bondary of 3D block of grid 
        // it is larger than actual block of data due to increase by size of the kernel
        // the data along the boundary of the block should not be used
        int bxmin, bymin, bzmin;
        int bsizex, bsizey,bsizez;  
        // if this is positive - the class will use attribute instead of state
        // and will assume, that 
        //  inside voxels have attribute value gridMaxAttributeValue an 
        //  outside voxles have attribute value 0  
        //  and linear interpolation on the boundary 
        //int gridMaxAttributeValue = 0;
        //double dGridMaxAttributeValue = 1.;
        DensityMaker m_densityMaker = new DensityMakerSubvoxel(1);

        boolean containsIsosurface = false;
        /**
           
         */
        public BlockSmoothingSlices(Grid grid){
            
            this.grid = (Grid)grid;
            if (grid instanceof AttributeGrid) {
                agrid = (AttributeGrid) grid;
            }
            grid.getGridBounds(this.bounds);

            gnx = grid.getWidth();
            gny = grid.getHeight();
            gnz = grid.getDepth();
            
            gdx = (bounds[1] - bounds[0])/(gnx-1);
            gdy = (bounds[3] - bounds[2])/(gny-1);
            gdz = (bounds[5] - bounds[4])/(gnz-1);

            gxmin = bounds[0];
            gymin = bounds[2];
            gzmin = bounds[4];

        }
        
        /**
           @obsolete            
         */
        public void setGridMaxAttributeValue(int value){

            m_densityMaker = new DensityMakerSubvoxel(value);
            
        }

        /**
           
           set DensityMaker which convertes from attribute value into density 
           density makes is supposed to return 
            0 - outside, 
            1 - inside
            0.5 on surface 
         */
        public void setDensityMaker(DensityMaker densityMaker){

            m_densityMaker = densityMaker;

        }
        /**
           
           block bounds are given in integer cordinates of voxels 
           block bounds are inclusive 
           if kernel is null - no convolution is performed 
         */
        public void initBlock(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax){ 
            initBlock(xmin, xmax, ymin, ymax, zmin, zmax, null);
        }
        public void initBlock(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax, double kernel[]){ 
            
            //printf("initBloc(%d %d %d %d %d %d)\n",xmin, xmax, ymin, ymax, zmin, zmax);
            
            int kernelSize = 0; 
            if(kernel != null){
                kernelSize = (kernel.length+1)/2;
            }

            bxmin = xmin - kernelSize;
            bymin = ymin - kernelSize;
            bzmin = zmin - kernelSize;

            bsizex = (xmax - xmin + 1) + 2*kernelSize;
            bsizey = (ymax - ymin + 1) + 2*kernelSize;
            bsizez = (zmax - zmin + 1) + 2*kernelSize;            
            
            int dataSize = bsizex * bsizey * bsizez;
            
            if(blockData == null || dataSize > blockData.length){
                blockData = new double[dataSize];
            }
            
            int maxSize = bsizex;
            if(bsizey > maxSize)maxSize = bsizey;
            if(bsizez > maxSize)maxSize = bsizez;
            
            if(rowData == null || rowData.length < maxSize)
                rowData = new double[maxSize];
            
            boolean hasPlus = false, hasMinus = false;
            
            // fill block with data from grid 
            for(int y = 0; y < bsizey; y++){
                int y0 = y + bymin;
                int xoffset  = y*bsizez*bsizex;

                for(int x = 0; x < bsizex; x++){
                    int x0 = x + bxmin;
                    int zoffset  = xoffset + x*bsizez;

                    for(int z = 0; z < bsizez; z++){
                        // TODO get data for z row in one call 
                        double v = getGridData(x0, y0, z + bzmin);
                        if(v > 0.)
                            hasPlus = true;
                        else if(v < 0.)
                            hasMinus = true;
                        blockData[zoffset + z] = v;
                        
                    }
                }
            } 
            
            if(hasPlus && hasMinus){

                containsIsosurface = true;
                if(kernelSize > 0){
                    convoluteX(blockData, kernel);
                    convoluteY(blockData, kernel);
                    convoluteZ(blockData, kernel);              
                }
            } else {
                containsIsosurface = false;
            }
            //printf("containsIsosurface:%s\n", containsIsosurface);

        }

        void convoluteX(double data[], double kernel[]){
            
            int ksize = kernel.length/2;
            int bsizexz = bsizex*bsizez;
            
            for(int y = 0; y < bsizey; y++){

                int offsetx = y * bsizexz;
                
                for(int z = 0; z < bsizez; z++){

                    // init accumulator array 
                    Arrays.fill(rowData, 0, bsizex, 0.);

                    for(int x = 0; x < bsizex; x++){
                        
                        int offsetz = offsetx + x * bsizez;
                        double v = blockData[offsetz + z];
                        // add this value to accumulator
                        for(int k = 0; k < kernel.length; k++){
                            int kx = x + k - ksize;
                            if(kx >= 0 && kx < bsizex)
                                rowData[kx] += kernel[k] * v;                            
                        }
                    } 
                    
                    for(int x = 0; x < bsizex; x++){
                        int offsetz = offsetx + x * bsizez;
                        blockData[offsetz + z] = rowData[x];
                    }                    
                }
            }
        }

        void convoluteY(double data[], double kernel[]){

            int ksize = kernel.length/2;
            int bsizexz = bsizex*bsizez;
            
            for(int x = 0; x < bsizex; x++){
                for(int z = 0; z < bsizez; z++){
                    // init accumulator array 
                    Arrays.fill(rowData, 0, bsizey, 0.);
                    for(int y = 0; y < bsizey; y++){
                        
                        int offsetz = y * bsizexz + x * bsizez;
                        double v = blockData[offsetz + z];
                        // add this value to accumulator
                        for(int k = 0; k < kernel.length; k++){
                            int ky = y + k - ksize;
                            if(ky >= 0 && ky < bsizey)
                                rowData[ky] += kernel[k] * v;                            
                        }
                    } 

                    for(int y = 0; y < bsizey; y++){
                        int offsetz = y * bsizexz + x * bsizez;
                        blockData[offsetz + z] = rowData[y];
                    }                    
                }
            }

        }

        void convoluteZ(double data[], double kernel[]){
            
            int ksize = kernel.length/2;
            int bsizexz = bsizex*bsizez;
            for(int y = 0; y < bsizey; y++){

                int offsetx = y * bsizexz;
                
                for(int x = 0; x < bsizex; x++){

                    int offsetz = offsetx + x * bsizez;
                    // init accumulator array 
                    Arrays.fill(rowData, 0, bsizez, 0.);

                    for(int z = 0; z < bsizez; z++){
                        
                        double v = blockData[offsetz + z];
                        // add this value to accumulator
                        for(int k = 0; k < kernel.length; k++){
                            int kz = z + k - ksize;
                            if(kz >= 0 && kz < bsizez)
                                rowData[kz] += kernel[k] * v;                            
                        }
                    } 
                    for(int z = 0; z < bsizez; z++){
                        blockData[offsetz + z] = rowData[z];
                    }                    
                }
            }
        }
        
        /**
           
         */
        public boolean containsIsosurface(){
            return containsIsosurface; 
        }

        static final int round(double x){
            return (int)Math.floor(x + 0.5);
        }

        public void getSlice(SliceData sliceData){

            int nx = sliceData.nx;
            int ny = sliceData.ny;

            double xmin = sliceData.xmin;
            double ymin = sliceData.ymin;

            double dx = (sliceData.xmax - xmin)/(nx-1);
            double dy = (sliceData.ymax - ymin)/(ny-1);            

            double z = sliceData.z;

            int gz = round((z - gzmin)/gdz);
            
            double data[] = sliceData.data;
            //try {
           
            for(int iy = 0; iy < ny; iy++){

                double y = ymin + iy*dy;
                int gy = round((y - gymin)/gdy);

                int offset = iy * nx;

                for(int ix = 0; ix < nx; ix++){

                    double x = xmin + ix*dx;

                    int gx = round((x - gxmin)/gdx);
                    data[offset + ix] = getBlockData(gx,gy,gz);
                }
            }           

            //} catch(Exception e){
            //    printf("problem: %d x %d data: %d thread:%s\n", nx, ny, data.length, Thread.currentThread().getName());
            //} 
 
        }

        /**
           return data at the grid point 
           does averaging
         */
        double getBlockData(int gx, int gy, int gz){

            gx -= bxmin;
            gy -= bymin;
            gz -= bzmin;
            
            if(gx < 0 || gx >= bsizex || gy < 0 || gy >= bsizey || gz < 0 || gz >= bsizez )
                return 1.;
                        
            return blockData[(gy*bsizex + gx) * bsizez + gz];
            
        }


        double getGridData(int gx, int gy, int gz){

            if(gx <  0 || gy < 0 || gz < 0 || gx >= gnx || gy >= gny || gz >= gnz){
                return 1.; // outside
            } else {
                
                // normalize output to interval (-1, 1) 
                // -1 - inside 
                // 1 - outside
                
                return 1-2*m_densityMaker.makeDensity(agrid.getAttribute(gx,gy,gz));

            }
        }
    } // class BlockSmoothingSlices
    
}
