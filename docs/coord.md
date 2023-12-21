# Notes on Coordinate System and Notations

Inferring coordinate systems (interchangeably referred as frames) from code can often be perplexing. Code frequently contains numerous normalizations, rigid transformations, inversions, and various operations scattered throughout. When I navigate through these lengthy lines of code, it can be challenging to discern the coordinate frames in use. To alleviate this confusion, I've found the following practices to be particularly helpful:


1. **Always write entities (points / triangles / meshes / transformation) with respect to their coodinate systems.** 
 Instead of merely referencing an object as $O$, explicitly denote it in its coordinate system (such as world frame for an example) as $^wO$. Represent transformations from coordinate system $w$ to $c$ as $^cT_w$ or $T_{w\to c}$. The advantage of using the former notation $^bT_a$ is that it aligns well with left-product transformations. For instance, when transforming an object in the world coordinate system by the world-to-camera transformation, the subscript and superscript just checks out as follows: $^cX = ^cT_w ^wX$.

2. **Always write code the way you write equation.**  Now, let's express code using the same notation. For example, the world-to-camera transformation mentioned earlier can be written in code as `cObj = cTw @ wObj`. This approach enhances the readability of our code and ensures that we can easily trace the coordinate systems being used, even after tons of operations have been performed.

3. **Coordinate specification in this project**. The primary coordinate systems we used in this project include: 
    1) object centric frame (denoted as `o` in code), which is the canonical object frame defined in each datasets. 
    2) normalized object frame (`u`) that normalizes object to a unit sphere, where object SDF grid are precomputed. 
    3) hand frame (`h`) whose origin sits at the hand wrist without wrist rotation, i.e. the palm alwasy facing upwards. Its scale is meter.  
    4) normalized hand frame / joint frame (`j`) whose origin is located above the index knuckle. It's scale is 0.2m. This is the primary frame we work with. 
    
    To transfrom from one corodinate system to another.\:
    - `uTo`: `preprocess/make_sdf_grid.py:nom_to_unit_ball()`
    - `hTn`: `jutils.hand_utils.get_nTh()`
    - `oTh`: Datasets specify them correspondingly and we ue them to train G-HOP prior.  During video reconstruction, we initialize them to be identity matrix since there is no canonical object pose. 

