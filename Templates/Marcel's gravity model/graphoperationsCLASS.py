'''
NEED TO LEARN MORE ABOUT CLASSES... !


purpose: create graph from specifications of the user with 
generally used node properties used in supply chain models



Methods:
----------
directly operating on network
* createGraph - the graph is created with node attributes
    'actors','locations','products'
* getlist_node_ids - returns a list of nodeIDs with the corresponding
    attributes, used for editing information on the graph
* getNodeIDswAttr - returns a tuple of (nodeIDs,attributes) with the
    corresponding attributes, used for editing information on the graph
* addAttr2ExistingNodes - adds a user specified attribute to the nodes
    of the network
* addAttr2Edges - adds a user specified attribute to the edges of the network
* getEdgeID - returns the IDs for the edges on the graph, given the user
    specified nodes (IDs given by getlist_node_ids) on the network
* getEdgesAttr - returns a list of attributes of the edges
    (IDs given by getEdgeID) specified by the user
* getEdgeIDswAttr - returns a tuple of
    ((edge ID, attribute) = (node out,node in, attribute))
    of the edges (IDs given by getEdgeID) specified by the user
* getExistingAttrs - returns a list of existing attributes on the graph
    (either nodes or edges)

toolkit for handling attributes on the network
* combineActorBrand - if an actor consists of different subtypes
    (e.g. warehouse and brand(ALDI,REWE, etc.)), then this function helps
    with a fast way of creating additional actors, before creating the network
* convertLoT2NTs - functions like getEdgeIDswAttr, return a list of
    tuples, but only a part of this is needed. This function accepts a list
    of tuples as input and returns 2 lists with the same order as before.
* convertTup2LoT - functions like addAttr2ExistingNodes, accept a list
    of tuples as input, but only a list available. This function accepts
    2 lists as input and returns a list on tuples with the same order as
    before.
* getAllCombinations - getEdgeID is a function that has the corresponding
    nodes as input, and returns a direct  1-to-1 list (node out, node in).
    If all possible combinations of edges between 2 nodes are to be
    returned, this function helps with that.
* convertMatrix - 2 applications:
    1. for calibration might be needed if calibration measure is
    of different dimension as the calculated flows
    2. if a matrix with redundant information needs to be created
    from a matrix with minimum information (e.g. similar geographical
    distances between nodes to be put into a distance matrix)

mapping raw data objects to network data objects
* proxyModel - simple fraction model. If data for correlated variable
    is available, this model may be used to use the proxy data to model
    the data in question
    (e.g. modellled data = produced eggs, proxy data= number of laying hens)
* getDistMatrix - accesses the graph for attribute 'distance' for
    user-specified nodes, return a dataframe containing information of the
    geographical distance between the locations of given nodes (needs to be
    entered into the graphe beforehand via the function addAttr2Edges)
    NOTE that the attribute 'distance' needs to be defined on the edges
    of the participating nodes.
* optFlow, minOptFlowForBeta - calculating and minimizing the optimal
    flow given supply and demand of a product and the distance between
    participating nodes. Needed for calculating minimum transport
    distance (see calcPotTransportDistances).
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
* propFlow - calculating the proportional flow given supply and demand
    of a product and the distance between participating nodes. Needed for
    calculating maximum transport distance (see calcPotTransportDistances).
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
* furnessModel - given geographical distance, supply and demand of
    the participating nodes and the free parameter (resistance factor)
    beta this function returns the flow between given nodes
    (beta needs to be determined by hymanModel / calibration)
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
        demanding the product.
* calcPotTransportDistances - calculates the range of possible average
    transport distances and returns a list of values within this interval.
    The number of returned values from this interval may be specified by
    the user (needed for hymanModel to determine beta for furnessModel).
    NOTE that each of the returned values are POSSIBLE average
    transport distance. The best one still needs to be determined(see
    calibration).
    NOTE that the attribute 'distance' needs to be defined on the
        edges of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
        demanding the product.
* hymanModel - calculates resistance factor beta given a transport
    distance. Returns the flow between participating nodes of supply and
    demand.
    NOTE that each of the returned values are POSSIBLE average transport
        distance. The best one still needs to be determined(see calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
        demanding the product.
* getWeightedDist - returns the average transport distance, given the flow
    between participating nodes of supply and demand.
    NOTE that each of the returned values are POSSIBLE average
        transport distance. The best one still needs to be determined(see
        calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
        demanding the product.
* calibration - function for calculating the best configuration of given
    transport distances and corresponding flows. A number of possible
    average transport distances are given by the function
    calcPotTransportDistances. The function hymanModel calculates
    the corresponding flows.
    Now this needs to be calibrated against a measure to determine which
    average transport distance fits best the calibration measure.
    This function accepts a 4D-Tensor of all flows as input
    [supply,demand,product,transport distance] and the calibration measure
    in question.
    NOTE that each of the returned values are POSSIBLE average
        transport distance. The best one still needs to be determined(see
        calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges
        of the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
        supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
        demanding the product.

* calibrationOLD - deprecated function for calculating the best
    configuration of given transport distances and corresponding flows.
    MCMC simulation. Only here for reasons of nostalgia.




===============================================================================
'''

import networkx as nx
from networkx.classes.digraph import DiGraph
# from networkx.classes.graph import Graph
import numpy as np
from itertools import chain
from ipfn import ipfn
from scipy import optimize
import warnings

warnings.filterwarnings("ignore")


class SCgraph(DiGraph):
    """
    purpose: create graph from specifications of the user with
    generally used node properties used in supply chain models



    Methods:
    ----------
    directly operating on network
    * createGraph - the graph is created with node attributes
    'actors','locations','products'
    * getlist_node_ids - returns a list of nodeIDs with the corresponding
        attributes, used for editing information on the graph
    * getNodeIDswAttr - returns a tuple of (nodeIDs,attributes) with the
        corresponding attributes, used for editing information on the graph
    * addAttr2ExistingNodes - adds a user specified attribute to the nodes
        of the network
    * addAttr2Edges - adds a user specified attribute to the edges of the
        network
    * getEdgeID - returns the IDs for the edges on the graph, given the
        user specified nodes (IDs given by getlist_node_ids) on the network
    * getEdgesAttr - returns a list of attributes of the edges
        (IDs given by getEdgeID) specified by the user
    * getEdgeIDswAttr - returns a tuple of
        ((edge ID, attribute) = (node out,node in, attribute))
        of the edges (IDs given by getEdgeID) specified by the user
    * getExistingAttrs - returns a list of existing attributes on the
        graph (either nodes or edges)

    toolkit for handling attributes on the network
    * combineActorBrand - if an actor consists of different subtypes
        (e.g. warehouse and brand(ALDI, REWE, etc.)), then this function
        helps with a fast way of creating additional actors, before creating
        the network
    * convertLoT2NTs - functions like getEdgeIDswAttr, return a list of
        tuples, but only a part of this is needed. This function accepts
        a list of tuples as input and returns 2 lists with the same order as
        before.
    * convertTup2LoT - functions like addAttr2ExistingNodes, accept a list of
        tuples as input, but only a list available. This function accepts 2
        lists as input and returns a list on tuples with the same order as
        before.
    * getAllCombinations - getEdgeID is a function that has the corresponding
        nodes as input, and returns a direct  1-to-1 list (node out, node in).
        If all possible combinations of edges between 2 nodes are to be
        returned, this function helps with that.
    * convertMatrix - 2 applications:
        1. for calibration might be needed if calibration measure is of
            different dimension as the calculated flows
        2. if a matrix with redundant information needs to be created from
            a matrix with minimum information (e.g. similar geographical
            distances between nodes to be put into a distance matrix)

    mapping raw data objects to network data objects
    * proxyModel - simple fraction model. If data for correlated variable
        is available, this model may be used to use the proxy data to model
        the data in question (e.g. modellled data = produced eggs,
        proxy data= number of laying hens)
    * getDistMatrix - accesses the graph for attribute 'distance'
        for user-specified nodes, return a dataframe containing information
        of the geographical distance between the locations of given nodes
        (needs to be entered into the graphe beforehand via the
        function addAttr2Edges)
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
    * optFlow, minOptFlowForBeta - calculating and minimizing the optimal
        flow given supply and demand of a product and the distance between
        participating nodes. Needed for calculating minimum transport
        distance (see calcPotTransportDistances).
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the nodes
            supplying the product.
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
    * propFlow - calculating the proportional flow given supply and demand
        of a product and the distance between participating nodes. Needed
        for calculating maximum transport distance
        (see calcPotTransportDistances).
        NOTE that the attribute 'distance' needs to be defined on the
            edges of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the
            nodes supplying the product.
        NOTE that the attribute 'distance' needs to be defined on the
            edges of the participating nodes.
    * furnessModel - given geographical distance, supply and demand of the
        participating nodes and the free parameter (resistance factor)
        beta this function returns the flow between given nodes
        (beta needs to be determined by hymanModel / calibration)
        NOTE that the attribute 'distance' needs to be defined on the
            edges of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the
            nodes supplying the product.
        NOTE that the attribute 'input' needs to be defined on the
            nodes demanding the product.
    * calcPotTransportDistances - calculates the range of possible
        average transport distances and returns a list of values within
        this interval. The number of returned values from this interval may
        be specified by the user (needed for hymanModel to determine
        beta for furnessModel).
        NOTE that each of the returned values are POSSIBLE average
        transport distance. The best one still needs to be determined
        (see calibration).
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the nodes
            supplying the product.
        NOTE that the attribute 'input' needs to be defined on the nodes
            demanding the product.
    * hymanModel - calculates resistance factor beta given a transport
        distance. Returns the flow between participating nodes of supply
        and demand.
        NOTE that each of the returned values are POSSIBLE average transport
            distance. The best one still needs to be determined(see calibration).
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the nodes
            supplying the product.
        NOTE that the attribute 'input' needs to be defined on the nodes
            demanding the product.
    * getWeightedDist - returns the average transport distance, given the
        flow between participating nodes of supply and demand.
        NOTE that each of the returned values are POSSIBLE average
            transport distance. The best one still needs to be determined
            (see calibration).
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the nodes
            supplying the product.
        NOTE that the attribute 'input' needs to be defined on the nodes
            demanding the product.
    * calibration - function for calculating the best configuration of given
        transport distances and corresponding flows. A number of possible
        average transport distances are given by the function
        calcPotTransportDistances. The function hymanModel calculates
        the corresponding flows.
        Now this needs to be calibrated against a measure to determine
        which average transport distance fits best the calibration measure.
        This function accepts a 4D-Tensor of all flows as input
        [supply,demand,product,transport distance]
        and the calibration measure in question.
        NOTE that each of the returned values are POSSIBLE average
        transport distance. The best one still needs to be determined(see
        calibration).
        NOTE that the attribute 'distance' needs to be defined on the edges
            of the participating nodes.
        NOTE that the attribute 'output' needs to be defined on the nodes
            supplying the product.
        NOTE that the attribute 'input' needs to be defined on the nodes
            demanding the product.

    * calibrationOLD - deprecated function for calculating the best
        configuration of given transport distances and corresponding flows.
        MCMC simulation. Only here for reasons of nostalgia.
    """

    myAttributes = ['actors', 'locations', 'products']

    def __init__(
            self,
            actors: list,
            locations: list,
            products: list,
            *args,
            **kwargs
    ):
        """
        purpose: create a directed graph with standard attributes for supply chain modelling:
        on creation of the graph a number of nodes are created based on the input lists.
        The nodes contain all possible combinations of the elements from the input list.
        The attributes are: 'actor','location','product'

        :param actors: a list of participating actors (list of string)
        :param locations: a list of locations involved (e.g. Landkreise in Germany)
        :param products: a list of products being transported (e.g. vegetables, eggs, ...)

        :return: Graph with nodes containing all possible combinations of actors, locations
        and products

        example:
        from supplychainmodulator import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']

        myNW = go.createGraph(listOfActors=act,listOfLocations=loc,\\
            listOfProducts=prod)




        =======================================================================
        """
        super(SCgraph, self).__init__(*args, **kwargs)
        check = CheckUserInput()
        check.is_list(actors)
        check.is_list(products)
        check.is_list(locations)

        self.actors = actors
        self.products = products
        self.locations = locations

        # create names for node attributes
        # (currently hardcoded, since this never changes (as of this moment))
        actor = self.myAttributes[0]
        location = self.myAttributes[1]
        product = self.myAttributes[2]

        '''
        go through all permutations and create nodes with 
        these 3 attribute names and a list of the 
        corresponding 3 attribute contents
        '''
        nodes = zip(
            range(
                len(self.actors) * len(self.locations) * len(self.products)
            ),
            [
                {
                    actor: al,
                    product: pl,
                    location: ll
                }
                for al in self.actors
                for pl in self.products
                for ll in self.locations
            ]
        )
        self.add_nodes_from(list(nodes))

    @staticmethod
    def combine_actor_brand(
            actor: str,
            brand: list
    ) -> list:
        """
        purpose: if an actor has a brand, but is the same type, this can be used
        to give it the correct designation
        (e.g. retailer: Warehouse_ALDI or Store_ALDI a.s.o.)
        usually such an actor type has multiple brands

        :param actor: an element of the actor list
        :param brand: a list of brands for different actors: actor_suffix

        :return: a combined list

        example:
        from supplychainmodelhelper import graphoperations as go

        act = ['producer','consumer','warehouse','store']

        newActors = go.combineActorBrand(actor='warehouse',brand=['ALDI','REWE','LIDL'])

        act.remove('warehouse')
        act.extend(newActors)

        print(act)


        ===============================================================================================
        """
        # TODO: sanity check if given actor is in graph
        # TODO: remove actor from graph, add new actors to graph

        combined_list = [actor + "_" + go for go in brand]
        return combined_list

    def get_list_node_ids(
            self,
            actors=None,
            products=None,
            locations=None
    ) -> list:
        """
        purpose: You provide an existing supply chain network and list of actors,
        products and locations, this function returns a list of node ids with
        these attributes.
        If an attribute is missing, this function assumes the user wants all nodes
        of the missing attribute.
        If an empty list is given, this function assumes the user searches for
        nodes that dont have this particular attribute. This should result in an
        Exception if zero nodes are found with not one of the properties.

        NOTE that the returned list of node ids is ordered as the original list of
        attributes given at SC creation.


        :param actors: (optional) a list of actors
        :param products: (optional) a list of products
        :param locations: (optional) a list of locations

        :return: a list of node IDs filtered to the specifications of the user

        example:
        from supplychainmodelhelper import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']
        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )

        producerIDs = go.getlist_node_ids(SCGraph=myNW, actors=['producer'])



        ===========================================================================
        """
        if actors is None:
            actors = list(
                set(
                    val
                    for val in
                    nx.get_node_attributes(self, 'actors').values()
                )
            )
        if products is None:
            products = list(
                set(
                    val for val in nx.get_node_attributes(
                        self,
                        'products'
                    ).values()
                )
            )
        if locations is None:
            locations = list(
                set(
                    val for val in nx.get_node_attributes(
                        self,
                        'locations'
                    ).values()
                )
            )
        check = CheckUserInput()
        check.is_list(actors)
        check.is_list(products)
        check.is_list(locations)

        # actual code for getting the correct IDs
        myDict = dict(self.nodes(data=True))
        listOfNodes = [
            n for n in myDict
            if myDict[n]['actors'] in actors
               and myDict[n]['locations'] in locations
               and myDict[n]['products'] in products
        ]

        # if restricting the list of Nodes leads to an empty list
        if len(listOfNodes) == 0:
            testloNA = bool(
                [n for n in myDict if myDict[n]['actors'] in actors]
            )
            testloNP = bool(
                [n for n in myDict if myDict[n]['products'] in products]
            )
            testloNL = bool(
                [n for n in myDict if myDict[n]['locations'] in locations]
            )

        return (listOfNodes)

    def add_attr_to_existing_nodes(
            self,
            list_node_ids: list,
            name_attr: str,
            list_attr: list
    ) -> bool:
        """
        purpose: add a custom attribute to a specified list of node IDs to your
        supply chain nodes with the correct IDs
    
        :param SCGraph: the graph created with
        createGraph(listOfActors: list, listOfLocations: list,
        listOfProducts: list)
        :param list_node_ids:  should be a list of the nodes you want to add your
        attribute to (if set to 'None', this function assumes, you want to give
        all existing nodes this attribute)
        :param name_attr: should be a recognizable string
        (e.g. 'outgoing', 'incoming', ...)
        :param list_attr:should be the list of values given to the chosen
        attributes name (length of this list should be equal to list_node_ids)
    
        :return: Boolean: True if all nodeIDs with corresponding attribute values
        are added to the graph
    
        example:
        from supplychainmodelhelper import graphoperations as go
    
        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']
        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )
    
        producerIDs = go.getlist_node_ids(myNW, actors=['producer'])
        listOfAGS = [
            '01111',
            '01112',
            '01113',
            '01111',
            '01112',
            '01113',
            '01111',
            '01112',
            '01113'
        ]
        didItWork = go.addAttr2ExistingNodes(
           SCGraph=myNW,
           list_node_ids=producerIDs,
           name_attr='ags',
           list_attr=listOfAGS
        )
        print('Is it stored into the graph: '+str(didItWork))
    
    
    
        ===========================================================================
        """
        done = False
    
        # if no list is given, assume user wants all nodes
        if list_node_ids is None:
            list_node_ids = list(dict(self.nodes(data=True)).keys())
    
        # make sure both input lists are, indeed, lists
        if not isinstance(list_node_ids, list):
            raise Exception('Node IDs ist not a list!')
        if not isinstance(list_attr, list):
            raise Exception('Attributes ist not a list!')
    
        # should be of the same length!
        if len(list_node_ids) == len(list_attr):
            done = True
        else:
            raise Exception(
                'length of list of node ids is not equal '
                'to length of list of attributes'
            )
    
        # TODO add attribute to graph and make sure the order is preserved....
        # assume both lists are ordered correctly
        nodes = zip(list_node_ids, [{name_attr: al} for al in list_attr])
        self.add_nodes_from(list(nodes))
    
        return done

    def get_node_ids_attr(
            self,
            name_attr: str,
            list_node_ids=None
    ):
        """
        TODO test, check if nodes exists?
        purpose: get list of tuple with (nid,content of nodes with nid)
        if no list of nodes are given, all nodes with this attribute are returned

        :param name_attr: name of the existing attribute
        :param list_node_ids: (optional)give a prepared list of node IDs

        :return: list of tuples: (node id, attribute)

        example:
        from supplychainmodelhelper import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']
        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )

        producerIDs = go.getListOfNodeIDs(SCGraph=myNW, actors=['producer'])
        print(go.getNodeIDswAttr(
                SCGraph=myNW,
                nameOfAttr='location',
                listOfNodeIDs=producerIDs
            )
        )



        ===========================================================================
        """
        # if no list is given, assume user wants all nodes
        if list_node_ids is None:
            list_node_ids = list(dict(self.nodes(data=True)).keys())

        # make sure attribute exists already!
        all_attr_so_far = self.get_existing_attrs(gtype='nodes')
        if name_attr not in all_attr_so_far:
            raise Exception(
                'Attribute given is not added to '
                'any node in the graph, please check!'
            )

        # make sure input list is, indeed, a list
        is_node_ids_list = isinstance(list_node_ids, list)
        if not is_node_ids_list:
            raise Exception('Node IDs ist not a list!')

        # get all nodes
        attr = nx.get_node_attributes(self, name_attr).items()
        my_list_of_tuples = [val for val in attr if val[0] in list_node_ids]

        return my_list_of_tuples

    def get_existing_attrs(
            self,
            gtype='n'
    ):
        """
        purpose: return a list of attributes that exist in this graph

        possible are nodes: 'nodes' or 'n'
        possible are edges: 'edges' or 'e'

        :param gtype: (optional) default is 'nodes'

        :return: a list of attributes that the user may access
        (after initialising: standard attributes are:
        'actor','location','product')

        example:
        from supplychainmodelhelper import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']
        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )

        print(go.getExistingAttrs(SCGraph=myNW, gtype = 'nodes'))
        print(go.getExistingAttrs(SCGraph=myNW, gtype = 'edges'))

        ===========================================================================
        """

        if gtype == 'nodes' or gtype == 'n':
            list_attributes = list(
                set(
                    chain(
                        *[list(x.keys())
                          for x in dict(self.nodes(data=True)).values()]
                    )
                )
            )
        elif gtype == 'edges' or gtype == 'e':
            list_attributes = list(
                set(
                    chain(*[list(x.keys())
                            for x in [
                                w for _, _, w in
                                list(self.edges(data=True))]
                            ]
                          )
                )
            )
        else:
            print('Error: This input was confusing, please try again')
            list_attributes = []

        return list_attributes

    @staticmethod
    def convert_list_tuples_to_n_tuples(
            list_tuples: list
    ):
        """
        purpose: convert/unpack a list of 2-tuples to two tuple lists
        usage: x,y = convertLoT2NTs(thelistoftuples)
        for turning it back, see: convertTup2LoT

        :param list_tuples: a list of tuples

        :return: depending on the size of the tuple, a number of lists are returned
        If a list of 2-tuples are given, 2 lists are returned.
        If a list of 3-tuples are given, 3 lists are returned.

        example:
        from supplychainmodelhelper import graphoperations as go
        prod = ['milk', 'beer', 'schnaps']
        act = ['producer', 'consumer', 'warehouse', 'store']
        loc = ['BER', 'SXF', 'TXL']

        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )

        senderIDs = go.getListOfNodeIDs(
            myNW,
            actors=['producer'],
            products=['milk']
        )
        receiverIDs = go.getListOfNodeIDs(
            SCGraph=myNW,
            actors=['consumer'],
            products=['milk']
        )

        myEdges4Graph = go.getEdgeID(
            SCGraph=myNW,
            outgoingNodes=senderIDs,
            incomingNodes=receiverIDs
        )
        list1, list2 = go.convertLoT2NTs(myEdges4Graph)



        ===========================================================================
        """
        return list(map(list, list(zip(*list_tuples))))

    @staticmethod
    def convert_n_tuples_to_list_tuples(
            tuple_1,
            tuple_2
    ):
        """
        TODO: test
        purpose: convert 2 lists or 2 tuples into 1 list of tuples
        if first argument is list of tuples, the return will be a list of 3 tuples
        usage;
        x = convertTup2LoT(tuple or list ,tuple or list)                or
        x = convertTup2LoT(list of tuples, tuple or list )
        for turning back, see convertLoT2NTs

        :param tuple_1: tuple or list
        :param tuple_2: tuple or list

        :return: a list of tuples

        example:
        from supplychainmodelhelper import graphoperations as go
        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']

        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )

        senderIDs = go.getListOfNodeIDs(
            SCGraph=myNW,
            actors=['producer'],
            products=['milk']
        )

        someContent = [2,4,2]
        contentConnected2NodeID = go.convertTup2LoT(senderIDs,someContent)


         ==========================================================================
        """
        if type(tuple_1) == list \
                and all(isinstance(i, tuple) for i in tuple_1):
            t1o, t1i = list(zip(*tuple_1))
            return list(zip(t1o, t1i, tuple_2))
        else:
            return list(zip(tuple_1, tuple_2))

    def proxy_model(
            self,
            input_float: float,
            proxy_model_data_list: list,
            proxy_float: float
    ) -> list:
        """
        purpose: This model maps raw data onto the graph via proxy data.
        A proportional equation is solved:
        networkData / inputNr = proxyData / proxyData.

        example:
        If you have the national production of eggs and want to map this
        on your network model with a location resolution on a county level and
        you have the data for how many chicken live in each county, you could come
        up with a proxyModel saying the number of chicken is proportional to the
        eggs produced in this county.
        Then:
        inputNr is the national production of eggs(float), proxyData is the number
        of chicken in each county(list) and proxyNr is the sum of all chicken in
        all counties. The returned list is then the modelled production of eggs
        in each county.

        Shorter version of this example:
        inputNr = national production of eggs
        proxyData = list of tuples of (NodeID, number of chicken in each location)
             this can also be just a list, but you get a warning if you do that
        proxyNr = total number of chicken
        inputNr/ModelData = proxyNr/proxyData ->
        ModelData = proxyData/proxyNr*inputNr

        :param input_float: float. (usage see explanation above)
        :param proxy_model_data_list: list or List of tuples
        (if you want to connect your node IDs with the respective attribute).
        (usage see explanation above)
        :param proxy_float: float. (usage see explanation above)


        :return: list of modelled data or list of tuples of
        (NodeID, modelled data), depending on proxyData.

        example:
        from supplychainmodelhelper import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']
        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )
        senderIDs = go.getListOfNodeIDs(
            SCGraph=myNW,
            actors=['producer'],
            products=['milk']
        )


        nationalProduction = 100.
        proxynatProduction = 10.

        print('creating list for input for proxymodel')
        proxyregProduction = [2,4,2]

        regionalProduction = go.proxyModel(
            inputNr=nationalProduction,
            proxyData=proxyregProduction,
            proxyNr=proxynatProduction
        )
        print(regionalProduction)

        print('creating list of tuples for input for proxymodel')
        proxyregProdWithNodeIDs = go.convertTup2LoT(senderIDs, proxyregProduction)

        regionalProduction = go.proxyModel(
            inputNr=nationalProduction,
            proxyData=proxyregProdWithNodeIDs,
            proxyNr=proxynatProduction
        )
        print(regionalProduction)




        ===========================================================================
        """
        try:
            nid, pd = self.convert_list_tuples_to_n_tuples(
                proxy_model_data_list
            )
        except ValueError:
            oid, iid, pd = self.convert_list_tuples_to_n_tuples(
                proxy_model_data_list
            )
        except TypeError:
            pd = proxy_model_data_list
            print(
                'Warning: proxyModel expected a list of tuples, '
                'but does its job anyway. '
                '\nReturned a list of modelled data with no Node IDs!'
            )

        output_data = [el / proxy_float * input_float for el in pd]
        try:
            output_data = self.convert_list_tuples_to_n_tuples(
                nid,
                output_data
            )
        except NameError:
            try:
                output_data = list(zip(oid, iid, output_data))
            except UnboundLocalError:
                pass

        return output_data

    @staticmethod
    def get_all_combinations(
            list1: list,
            list2: list,
            order='1st'
    ):
        """
        purpose: put in 2 lists, get out all possible combinations of element pairs
        in 2 lists

        :param list1: list with X elements
        :param list2: list with Y elements
        :param order: (optional) str for ordering which list should iterate fast
        and which slow
        default '1st': elements of list1 run fast and elements of list2 run slow
        other '2nd': elements of list2 run fast and elements of list1 run slow

        :return: 2 lists with all pair-wise combinations

        example:
        from supplychainmodelhelper import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']

        myNewList1,myNewList2 = go.getAllCombinations(prod,act)

        print(myNewList1)
        #>> ['milk', 'beer', 'schnaps', 'milk', 'beer', 'schnaps', 'milk', \\
        #    'beer', 'schnaps', 'milk', 'beer', 'schnaps']
        print(myNewList2)
        #>> ['producer', 'producer', 'producer', 'consumer', 'consumer', \\
        #    'consumer', 'warehouse', 'warehouse', 'warehouse',
        #    'store', 'store', 'store']



        ===============================================================================================
        """
        if order == '1st':
            all_combinations1 = [o for _ in list2 for o in list1]
            all_combinations2 = [i for i in list2 for _ in list1]
        elif order == '2nd':
            all_combinations1 = [i for i in list1 for _ in list2]
            all_combinations2 = [o for _ in list1 for o in list2]
        else:
            raise Exception("please set the parameter \'order\' to either "
                            "\'1st\' or \'2nd\'")

        return all_combinations1, all_combinations2

    def get_edge_id(
            self,
            outgoing_nodes: list,
            incoming_nodes: list
    ):
        """
        purpose: get edge id from corresponding list of nodes to be connected
        Note this is a one-to-one connection: the first element of outgoingNodes is
        ONLY connected to the first element in incomingNodes. If you want to return
        a list of edgeIDs with all possible combinations, either create 2 lists of
        all possible combinations yourself or make use of
        myNewList1,myNewList2 = getAllCombinations(list1,list2)
        NOTE this function doesnt check if any of the edges already contain
        information


        :param SCGraph: the graph, needed for confirmation that the nodes to be
        connected do indeed exist
        :param outgoing_nodes: a list of NodeIDs of existing nodes in SCGraph
        (can be retrieved with getNodeID)
        :param incoming_nodes: a list of NodeIDs  of existing nodes in SCGraph
        (can be retrieved with getNodeID)

        :return: a list of tuples (outgoingNodes,incomingNodes)

        example:
        from supplychainmodelhelper import graphoperations as go

        prod = ['milk','beer','schnaps']
        act = ['producer','consumer','warehouse','store']
        loc = ['BER','SXF','TXL']

        myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
        )
        producerIDs = go.getListOfNodeIDs(SCGraph=myNW, actors=['producer'])
        consumerOfSchnapsIDs = go.getListOfNodeIDs(
            SCGraph=myNW,
            products=['schnaps'],
            actors=['producer']
        )
        allcomb1,allcomb2 = go.getAllCombinations(
            producerIDs,
            consumerOfSchnapsIDs
        )

        myEdgeIDs = go.getEdgeID(
            SCGraph=myNW,
            outgoingNodes=allcomb1,
            incomingNodes=allcomb2
        )

        ===========================================================================
        """
        complete_list_all_nodes_in_graph = [
            n 
            for n in dict(self.nodes(data=True)).keys()
        ]

        all_there = all(
            [
                all(
                    r in complete_list_all_nodes_in_graph 
                    for r in outgoing_nodes
                ),
                all(
                    r in complete_list_all_nodes_in_graph 
                    for r in incoming_nodes
                )
            ]
        )

        if all_there:
            list_of_edge_ids = self.convert_n_tuples_to_list_tuples(
                outgoing_nodes, 
                incoming_nodes
            )
        else:
            out_here = all(
                r in complete_list_all_nodes_in_graph for r in outgoing_nodes
            )
            in_here = all(
                r in complete_list_all_nodes_in_graph for r in incoming_nodes
            )
            if not out_here:
                raise Exception(
                    'found some nodes that are not existing! '
                    '(check outgoing nodes)'
                )
            if not in_here:
                raise Exception(
                    'found some nodes that are not existing! '
                    '(check incoming nodes)'
                )
        return list_of_edge_ids


class CheckUserInput:
    def __init__(self):
        self.everything_allright = True

    def is_list(self, current_list):
        if not isinstance(current_list, list):
            self.everything_allright = False
            raise Exception('actors ist not a list!')

        if len(current_list) == 0:
            self.everything_allright = False
            raise Exception(
                'incorrect parameter: '
                'this should be a list, but it is: %20s' % current_list
            )
