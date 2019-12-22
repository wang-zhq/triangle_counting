#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <sys/stat.h>
#include <parallel/algorithm>

using namespace std;

int64_t get_filesize(char const *filename)
{
    struct stat statbuf;
    int r = stat(filename, &statbuf);
    if (r == -1)
    {
        cout << "Get size of file is failed!" << endl;
        exit(-1);
    }
    return (statbuf.st_size);
}

int main(int argc, char const **argv)
{
    char const *dt_name;
    // 接收命令并读取文件
    if (argc < 1)
    {
        cout << "Usage: " << argv[0] << " filename." << endl;
        exit(1);
    }
    else if (argc == 3)
        dt_name = argv[2];
    else
        dt_name = argv[1];

    size_t dt_size = get_filesize(dt_name);

    cout << "The size of data file " << dt_name << " is " << dt_size << " Bytes." << endl;
    time_t t0 = time(NULL);

    // 读取文件可以有好多种方法
    // 第一种： fread
    FILE *fp = fopen(dt_name, "rb");
    if (fp == NULL)
    {
        cout << "Can't open " << dt_name << "." << endl;
        exit(1);
    } 

    uint32_t *endpoints = (uint32_t *)malloc(sizeof(char)*dt_size);

    // fread 不加变量也可以，但在编译时会有警告
    size_t fin = fread(endpoints, sizeof(char), dt_size, fp);
    fclose(fp);
    
    // 第二种： fstream
    // ifstream fin(dt_name, ios::binary);
    // fin.read((char*)endpoints, dt_size);
    // fin.close();

    // 第三种： open 和 mmap
    // int fp = open(dt_name, O_RDONLY);
    // if (fp == -1)
    // {
    //     cout << "Can't open " << dt_name << "." << endl;
    //     exit(1);
    // } 

    // 通过内存映射实现快速读取
    // uint32_t *endpoints;
    // endpoints = (uint32_t*)mmap(NULL, dt_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fp, 0);
    // if (endpoints == NULL || endpoints == (void*)-1)
    // {
    //     cout << "Mapping Failed!" << endl;
    //     close(fp);
    //     exit(-2);
    // }

    size_t n_edge = dt_size/8;
    cout << "Data is loaded successfully. There are " << n_edge << " edges." << endl; 
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl;

    cout << "Data preprocessing starts..." << endl;

    // 找到所有数里最大的那个，即为节点的总个数
    int64_t i, p;
    
    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        if (endpoints[i*2] < endpoints[i*2+1])
        {
            uint32_t tmp = endpoints[i*2];
            endpoints[i*2] = endpoints[i*2+1];
            endpoints[i*2+1] = tmp;
        }
    }

    // 由于数据载入需要较长时间，上面的步骤做与不做关系不大
    // 如果做了查找最大值工作量减半
    cout << "数据预览完成。" << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    uint32_t dim = 0;
    #pragma omp parallel for reduction(max:dim)
    for (i = 0; i < n_edge; i++)
        dim = max(endpoints[i*2], dim);

    dim++;

    cout << "顶点个数统计完毕，共有 " << dim << " 个顶点" << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    // 根据线程数对空间进行划分，先区间分别计算顶点连接数，避免原子操作
    // 如果剩余内存太小，也只能采用原子操作，这里以60G为内存上限进行设置
    int n_proc = omp_get_max_threads();
    int n_limit = (60/4*1024*1024 - n_edge*2/1024)/(max(dim/1024,dim/dim)) - 3;
    int n_zone = min(min(n_proc, n_limit), 8);

    uint32_t *freq_count = new uint32_t [dim*2] ();

    // 统计数据出现频次，并根据频次重新编号
    if (n_limit < 1)
    {
        #pragma omp parallel for
        for (i = 0; i < dim; i++)
            freq_count[i*2] = i;

        #pragma omp parallel for
        for (i = 0; i < n_edge*2; i++)
        {
            #pragma omp atomic
            freq_count[endpoints[i]*2+1]++;
        }
    }
    else
    {
        int64_t prc_size = n_edge/n_zone;
        
        uint32_t *freq_group = new uint32_t [n_zone*dim]();
        // #pragma omp parallel for
        // for (i = 0; i < n_zone*dim; i++)
        //     freq_group[i] = 0;

        #pragma omp parallel for
        for (p = 0; p < n_zone; p++)
        {
            int64_t pini = p*dim;
            int64_t zini = p*prc_size;

            size_t z_num;
            if (p == n_zone-1)
                z_num = n_edge - zini;
            else
                z_num = prc_size;

            for (int64_t k = zini; k < zini+z_num; k++)
                freq_group[pini+endpoints[k]]++;
        }
        
        #pragma omp parallel for
        for (i = 0; i < dim; i++)
        {
            freq_count[i*2] = i;
            freq_count[i*2+1] = freq_group[i];
            for (int16_t p = 1; p < n_zone; p++)
                freq_count[i*2+1] += freq_group[p*dim+i]; 
        }

        delete [] freq_group;
    }
    
    uint64_t *fcz = (uint64_t *)freq_count;
    __gnu_parallel::sort(fcz, fcz+dim);

    cout << "顶点连接度统计完成，最大连接度为 " << freq_count[dim*2-1] << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    // 经刘老师提醒修改，生成索引对照表， 与下面注释部分等效
    // #pragma omp parallel for
    // for (i = 0; i < dim; i++)
    //     freq_count[freq_count[i*2]*2+1] = i;
    
    // 如要替换成下面的部分，取值的位置为前点
    // #pragma omp parallel for
    // for (i = 0; i < dim; i++)
    // {
    //     freq_count[i*2+1] = freq_count[i*2];
    //     freq_count[i*2] = i;
    // }
    // __gnu_parallel::sort(fcz, fcz+dim);

    
    // 按新编号给原数据换血
    // #pragma omp parallel for
    // for (i = 0; i < n_edge*2; i++)
    //     endpoints[i] = freq_count[endpoints[i]*2+1];

    uint32_t *new_seri = new uint32_t [dim];

    #pragma omp parallel for
    for (i = 0; i < dim; i++)
        new_seri[freq_count[i*2]] = i; 
    
    delete [] freq_count;

    // 按新编号给原数据换血
    #pragma omp parallel for
    for (i = 0; i < n_edge*2; i++)
        endpoints[i] = new_seri[endpoints[i]];
    
    delete [] new_seri;

    cout << "按连接数重新编号完成！" << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    // 实际大端在前，小端在后
    #pragma omp parallel for
    for (i = 0; i < n_edge; i++)
    {
        if (endpoints[i*2] < endpoints[i*2+1])
        {
            uint32_t tmp = endpoints[i*2];
            endpoints[i*2] = endpoints[i*2+1];
            endpoints[i*2+1] = tmp;
        }
        else if (endpoints[i*2] == endpoints[i*2+1])
        {
            endpoints[i*2+1] = dim;
        }
    }
    
    // =============================
    // 对大型数据集按CPU核数进行分段排序
    uint64_t *ptrz0 = (uint64_t *)endpoints;

    if (n_edge > 4*1024*1024)
    {
        int64_t prc_size = (n_edge/n_proc)*n_proc;
        int n_slice = 1;      

        while (n_slice < n_proc)
        {
            prc_size /= 2;

            if (n_slice == 1)
                __gnu_parallel::nth_element(ptrz0, ptrz0+prc_size, ptrz0+n_edge);
            else if (n_slice == 2)
            {
                __gnu_parallel::nth_element(ptrz0, ptrz0+prc_size, ptrz0+prc_size*2);
                __gnu_parallel::nth_element(ptrz0+prc_size*2, ptrz0+prc_size*3, ptrz0+n_edge);
            }
            else
            {
                #pragma omp parallel for
                for (p = 0; p < n_slice; p++)
                {
                    int64_t pini, p_end;
                    pini = p*prc_size*2;
                    if (p == (n_slice-1))
                        p_end = n_edge;
                    else
                        p_end = (p+1)*prc_size*2;
                    
                    nth_element(ptrz0+pini, ptrz0+pini+prc_size, ptrz0+p_end);
                }
            }
            
            n_slice *= 2;
        }
        
        cout << "排序分块完成！" << endl;
        cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

        #pragma omp parallel for
        for (p = 0; p < n_slice; p++)
        {
            int64_t pini, p_end;
            pini = p*prc_size;

            if (p == (n_slice-1))
                p_end = n_edge;
            else
                p_end = pini + prc_size;
            
            sort(ptrz0+pini, ptrz0+p_end);
        }
    }
    else
        __gnu_parallel::sort(ptrz0, ptrz0+n_edge);
   
    cout << "并行排序完成！" << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    // 去重没有并行版，其实也很快了
    uint64_t *ptrzn = unique(ptrz0, ptrz0+n_edge);
    int64_t num = ptrzn - ptrz0;

    cout << "去重完成！" << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    // 建立行首地址表, 用指针形式
    // int64_t *row_start_pt = new int64_t [dim+1];
    uint32_t* *row_start_pt = new uint32_t* [dim+1]; 

    for (i = 0; i <= endpoints[0*2+1]; i++)
        row_start_pt[i] = endpoints;

    #pragma omp parallel for schedule(dynamic, 512)
    for (i = 1; i < num; i++)
    {
        if (endpoints[i*2+1] > endpoints[(i-1)*2+1])
        {
            for (int64_t k = (endpoints[(i-1)*2+1]+1); k <= endpoints[i*2+1]; k++) 
                row_start_pt[k] = endpoints + i;
        }
    }

    for (i = (endpoints[num*2-1]+1); i <= dim; i++)
        row_start_pt[i] = endpoints + num;
    
    // 有效边数以dim 行为界, dim行之后的内容无需考虑
    // num = row_start_pt[dim];

    cout << "处理后的有效边数为 " << num << endl;
    cout << "预处理全部完成！！" << endl;
    cout << "Time elapses " << time(NULL)-t0 << " sec." << endl << endl;

    // ===============
    // 把列序号移动前半部
    for (i = 1; i < num/4; i++)
        endpoints[i] = endpoints[i*2];

    #pragma omp parallel for
    for (i = num/4; i < num/2; i++)
        endpoints[i] = endpoints[i*2];
    
    #pragma omp parallel for
    for (i = num/2; i < num; i++)
        endpoints[i] = endpoints[i*2];

    // 上面的内容等效于：
    // for (i = 1; i < num; i++)
    //     endpoints[i] = endpoints[i*2];
    
    // 后半截空出来，就可以用它来存点东东了
    // 将一段内存清空，按CPU核数*dim进行
    memset(endpoints+n_edge, 0, (n_proc+4)*dim);
    // 之前不会用 memset
    // #pragma omp parallel for
    // for (i = n_edge; i < n_edge+n_proc*dim/4; i++)
    //     endpoints[i] = 0;

    int f_ratio = 0;

    // 有所有点对应的集合求交, 对填充率低的利用比较策略，对填充高的利用原子加
    #pragma omp parallel for schedule(dynamic, 4)
    for (i = 0; i < (dim-1); i++)
    {
        uint32_t nti = 0;
        
        uint32_t* r_st = row_start_pt[i];
        uint32_t* rl_n = row_start_pt[i+1];
        uint32_t v_rn = *(rl_n-1);
        
        if (rl_n-r_st <= 1)
            continue;
        
        if (f_ratio < (r_st-endpoints)*10/num)
        {
            f_ratio++;
            printf("计数已经完成 %ld%%.\n", (r_st-endpoints)*100/num);
        }

        uint32_t* c_loc;
        uint32_t* cl_n;

        // if ((rl_n-r_st)*100000/(dim-i+100) < 1)
        // 下面为求交集的三种方法，在不同的情况下表现不同
        // 但如何进行最优配比，还有待进一步研究
        // 两个有序集合求交集的一般做法
        if ((v_rn < (dim+i*3)/4) || (rl_n-r_st <= 16))
        {
            uint32_t* r_loc;
            
            for (uint32_t* k = r_st; k < (rl_n-1); k++)
            {
                c_loc = row_start_pt[*k];
                cl_n = row_start_pt[*k+1];

                r_loc = k + 1;
                while ((r_loc < rl_n) && (c_loc < cl_n))
                {
                    if (*c_loc > *r_loc)
                        r_loc++;
                    else if (*c_loc < *r_loc)
                        c_loc++;
                    else
                    {
                        nti++;
                        c_loc++;
                        r_loc++;
                    }
                }
            }
        }
        // 第一第三种的混合模式，用于两个集合首尾相差较大情况
        else if (v_rn < (dim*3+i)/4)
        {
            int proc_id = omp_get_thread_num();
            char *vl_ini = (char*)endpoints + 4*n_edge + proc_id*dim;
            
            uint32_t* k;
            uint32_t* z;
            uint32_t v_cn;
            for (k = r_st; k < rl_n; k++)
                *(vl_ini + *k) = 1;
            
            for (k = r_st; k < (rl_n-1); k++)
            {
                c_loc = row_start_pt[*k];
                cl_n = row_start_pt[*k+1];
                v_cn = *(cl_n-1);

                if (v_rn < v_cn)
                {
                    while (*c_loc < v_rn)
                        nti += *(vl_ini + *(c_loc++));
                }
                else
                {
                    for (z = c_loc; z < cl_n; z++)
                        nti += *(vl_ini + *z);
                }
                    
            }

            for (k = r_st; k < rl_n; k++)
                *(vl_ini + *k) = 0;
        }
        // 用于数据相对稠密的情况，把对比行按位展开，而后进行取位相加
        else
        {
            int proc_id = omp_get_thread_num();
            char *vl_ini = (char*)endpoints + 4*n_edge + proc_id*dim;

            uint32_t* k;
            uint32_t* z;
            for (k = r_st; k < rl_n; k++)
                *(vl_ini + *k) = 1;

            for (k = r_st; k < (rl_n-1); k++)
            {
                c_loc = row_start_pt[*k];
                cl_n = row_start_pt[*k+1];

                for (z = c_loc; z < cl_n; z++)
                    nti += *(vl_ini + *z);
            }

            for (k = r_st; k < rl_n; k++)
                *(vl_ini + *k) = 0;
        }

        // 把每一行的计数结果，记在映射空间的后面的空余位置
        endpoints[n_edge+n_proc*dim/4+i] = nti;
    }
    printf("计数 100%% 完成！\n");

    int64_t tri_ttl = 0;

    #pragma omp parallel for reduction(+:tri_ttl)
    for (i = (n_edge+n_proc*dim/4); i < (n_edge+n_proc*dim/4+dim-1); i++)
        tri_ttl += endpoints[i];

    delete [] row_start_pt;
    free(endpoints);

    cout << "There are \033[1m" << tri_ttl << "\033[0m triangles in the input graph." << endl;
    cout << "ALL DONE. Time elapses " << time(NULL)-t0 << " sec." << endl;

    return 0;
}
