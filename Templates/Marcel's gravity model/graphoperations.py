"""

purpose: create graph from specifications of the user with
generally used node properties used in supply chain models



Functions:
----------
directly operating on network
* createGraph - the graph is created with node attributes 'actors','locations',
'products'
* getListOfNodeIDs - returns a list of nodeIDs with the corresponding
attributes, used for editing information on the graph
* getNodeIDswAttr - returns a tuple of (nodeIDs,attributes) with the
corresponding attributes, used for editing information on the graph
* addAttr2ExistingNodes - adds a user specified attribute to the nodes of the
network
* addAttr2Edges - adds a user specified attribute to the edges of the network
* getEdgeID - returns the IDs for the edges on the graph, given the user
specified nodes (IDs given by getListOfNodeIDs) on the network
* getEdgesAttr - returns a list of attributes of the edges (IDs given by
getEdgeID) specified by the user
* getEdgeIDswAttr - returns a tuple of
((edge ID, attribute) = (node out,node in, attribute)) of the edges (IDs given
by getEdgeID) specified by the user
* getExistingAttrs - returns a list of existing attributes on the graph
(either nodes or edges)

toolkit for handling attributes on the network
* combineActorBrand - if an actor consists of different subtypes (e.g.
    warehouse and brand(ALDI,REWE, etc.)), then this function helps with a fast
    way of creating additional actors, before creating the network
* convertLoT2NTs - functions like getEdgeIDswAttr, return a list of tuples, but
    only a part of this is needed. This function accepts a list of tuples as
    input and returns 2 lists with the same order as before.
* convertTup2LoT - functions like addAttr2ExistingNodes, accept a list of
    tuples as input, but only a list available. This function accepts 2 lists
    as input and returns a list on tuples with the same order as before.
* getAllCombinations - getEdgeID is a function that has the corresponding nodes
    as input, and returns a direct  1-to-1 list (node out, node in). If all
    possible combinations of edges between 2 nodes are to be returned, this
    function helps with that.
* convertMatrix - 2 applications:
    1. for calibration might be needed if calibration measure is of different
        dimension as the calculated flows
    2. if a matrix with redundant information needs to be created from a matrix
        with minimum information (e.g. similar geographical distances between
        nodes to be put into a distance matrix)

mapping raw data objects to network data objects
* proxyModel - simple fraction model. If data for correlated variable is
    available, this model may be used to use the proxy data to model the data
    in question (e.g. modellled data = produced eggs, proxy data= number of
    laying hens)
* getDistMatrix - accesses the graph for attribute 'distance' for
    user-specified nodes, return a dataframe containing information of the
    geographical distance between the locations of given nodes (needs to be
    entered into the graph beforehand via the function addAttr2Edges)
    NOTE that the attribute 'distance' needs to be defined on the edges of the
    participating nodes.
* optFlow, minOptFlowForBeta - calculating and minimizing the optimal flow
    given supply and demand of a product and the distance between participating
    nodes. Needed for calculating minimum transport distance (see
    calcPotTransportDistances).
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'distance' needs to be defined on the edges of the
    participating nodes.
* propFlow - calculating the proportional flow given supply and demand of a
    product and the distance between participating nodes. Needed for
    calculating maximum transport distance (see calcPotTransportDistances).
    NOTE that the attribute 'distance' needs to be defined on the edges of the
    participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
* furnessModel - given geographical distance, supply and demand of the
    participating nodes and the free parameter (resistance factor) beta this
    function returns the flow between given nodes
    (beta needs to be determined by hymanModel / calibration)
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
    demanding the product.
* calcPotTransportDistances - calculates the range of possible average
    transport distances and returns a list of values within this interval.
    The number of returned values from this interval may be specified by the
    user (needed for hymanModel to determine beta for furnessModel).
    NOTE that each of the returned values are POSSIBLE average transport
    distance. The best one still needs to be determined(see calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
    demanding the product.
* hymanModel - calculates resistance factor beta given a transport distance.
    Returns the flow between participating nodes of supply and demand.
    NOTE that each of the returned values are POSSIBLE average transport
    distance. The best one still needs to be determined(see calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
    demanding the product.
* getWeightedDist - returns the average transport distance, given the flow
    between participating nodes of supply and demand.
    NOTE that each of the returned values are POSSIBLE average transport
    distance. The best one still needs to be determined(see calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
    demanding the product.
* calibration - function for calculating the best configuration of given
    transport distances
    and corresponding flows. A number of possible average transport
    distances are given by the function calcPotTransportDistances. The
    function hymanModel calculates the corresponding flows.
    Now this needs to be calibrated against a measure to determine which
    average transport distance fits best the calibration measure.
    This function accepts a 4D-Tensor of all flows as input [supply,demand,
    product,transport distance]
    and the calibration measure in question.
    NOTE that each of the returned values are POSSIBLE average transport
    distance. The best one still
    needs to be determined(see calibration).
    NOTE that the attribute 'distance' needs to be defined on the edges of
    the participating nodes.
    NOTE that the attribute 'output' needs to be defined on the nodes
    supplying the product.
    NOTE that the attribute 'input' needs to be defined on the nodes
    demanding the product.

* calibrationOLD - deprecated function for calculating the best
    configuration of given transport distances and corresponding flows. MCMC
    simulation. Only here for reasons of nostalgia.




===============================================================================
"""

import networkx as nx
from networkx.classes.digraph import DiGraph
from networkx.classes.graph import Graph
import numpy as np
from itertools import chain
from ipfn import ipfn
from scipy import optimize
import warnings

# warnings.filterwarnings("ignore")


def createGraph(
    listOfActors: list, listOfLocations: list, listOfProducts: list
) -> DiGraph:
    """
    purpose: create a directed graph with standard attributes for supply chain
    modelling:
    on creation of the graph a number of nodes are created based on the input
    lists. The nodes contain all possible combinations of the elements from
    the input list.
    The attributes are: 'actor','location','product'

    :param listOfActors: a list of participating actors (list of string)
    :param listOfLocations: a list of locations involved
    (e.g. Landkreise in Germany)
    :param listOfProducts: a list of products being transported
    (e.g. vegetables, eggs, ...)

    :return: Graph with nodes containing all possible combinations of actors,
    locations and products

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



    ===========================================================================
    """
    isItListA = isinstance(listOfActors, list)
    isItListP = isinstance(listOfProducts, list)
    isItListL = isinstance(listOfLocations, list)

    if not isItListA:
        raise Exception("actors ist not a list!")
    if not isItListP:
        raise Exception("products ist not a list!")
    if not isItListL:
        raise Exception("locations ist not a list!")

    howManyActors = len(listOfActors)
    howManyProducts = len(listOfProducts)
    howManyLocations = len(listOfLocations)

    if howManyActors == 0:
        raise Exception(
            "incorrect parameter: actors should be a list, "
            "but it is: %20s" % (listOfActors)
        )
    if howManyProducts == 0:
        raise Exception(
            "incorrect parameter: products should be a list, "
            "but it is: %20s" % (listOfActors)
        )
    if howManyLocations == 0:
        raise Exception(
            "incorrect parameter: locations should be a list, "
            "but it is: %20s" % (listOfActors)
        )

    DG = nx.DiGraph()

    # create names for node attributes (currently hardcoded,
    # since this never changes (as of this moment))
    actor = "actor"
    location = "location"
    product = "product"

    """
    go through all permutations and create nodes with these 3 attribute names 
    and a list of the corresponding 3 attribute contents
    """
    nodes = zip(
        range(len(listOfActors) * len(listOfLocations) * len(listOfProducts)),
        [
            {actor: al, product: pl, location: ll}
            for al in listOfActors
            for pl in listOfProducts
            for ll in listOfLocations
        ],
    )

    DG.add_nodes_from(list(nodes))

    return DG


def combineActorBrand(actor: str, brand: list) -> list:
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

    newActors = go.combineActorBrand(
        actor='warehouse',
        brand=['ALDI','REWE','LIDL']
    )

    act.remove('warehouse')
    act.extend(newActors)

    print(act)


    ===========================================================================
    """
    combinedList = [actor + "_" + go for go in brand]
    return combinedList


def getListOfNodeIDs(
    SCGraph: Graph, actors=None, products=None, locations=None
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


    :param SCGraph: the graph created with
    createGraph(listOfActors: list, listOfLocations: list,
    listOfProducts: list)
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

    producerIDs = go.getListOfNodeIDs(SCGraph=myNW, actors=['producer'])



    ===========================================================================
    """
    if actors == None:
        actors = list(
            set(val for val in nx.get_node_attributes(SCGraph, "actor").values())
        )
    if products == None:
        products = list(
            set(val for val in nx.get_node_attributes(SCGraph, "product").values())
        )
    if locations == None:
        locations = list(
            set(val for val in nx.get_node_attributes(SCGraph, "location").values())
        )

    isItListA = isinstance(actors, list)
    isItListP = isinstance(products, list)
    isItListL = isinstance(locations, list)

    if not isItListA:
        raise Exception("actors ist not a list!")
    if not isItListP:
        raise Exception("products ist not a list!")
    if not isItListL:
        raise Exception("locations ist not a list!")

    howManyActors = len(actors)
    howManyProducts = len(products)
    howManyLocations = len(locations)

    if howManyActors == 0:
        raise Exception(
            "incorrect parameter: actors should be a list, "
            "but it is: %20s" % (actors)
        )
    if howManyProducts == 0:
        raise Exception(
            "incorrect parameter: products should be a list, "
            "but it is: %20s" % (products)
        )
    if howManyLocations == 0:
        raise Exception(
            "incorrect parameter: locations should be a list, "
            "but it is: %20s" % (locations)
        )

    listOfNodes = []

    # actual code for getting the correct IDs
    myDict = dict(SCGraph.nodes(data=True))
    listOfNodes = [
        n
        for n in myDict
        if myDict[n]["actor"] in actors
        and myDict[n]["location"] in locations
        and myDict[n]["product"] in products
    ]

    # if restricting the list of Nodes leads to an empty list
    if len(listOfNodes) == 0:
        testloNA = bool([n for n in myDict if myDict[n]["actor"] in actors])
        testloNP = bool([n for n in myDict if myDict[n]["product"] in products])
        testloNL = bool([n for n in myDict if myDict[n]["location"] in locations])

        if not testloNA:
            raise Exception(
                "no nodes found with these conditions:  "
                "actors = %20s, length = %2d !)" % (actors, howManyActors)
            )
        if not testloNP:
            raise Exception(
                "no nodes found with these conditions: "
                "products = %20s, "
                "length = %2d !)" % (products, howManyProducts)
            )
        if not testloNL:
            raise Exception(
                "no nodes found with these conditions: "
                "locations = %20s, "
                "length = %2d !)" % (locations, howManyLocations)
            )

    return listOfNodes


def addAttr2ExistingNodes(
    SCGraph: Graph, listOfNodeIDs: list, nameOfAttr: str, listOfAttr: list
) -> bool:
    """
    purpose: add a custom attribute to a specified list of node IDs to your
    supply chain nodes with the correct IDs

    :param SCGraph: the graph created with
    createGraph(listOfActors: list, listOfLocations: list,
    listOfProducts: list)
    :param listOfNodeIDs:  should be a list of the nodes you want to add your
    attribute to (if set to 'None', this function assumes, you want to give
    all existing nodes this attribute)
    :param nameOfAttr: should be a recognizable string
    (e.g. 'outgoing', 'incoming', ...)
    :param listOfAttr:should be the list of values given to the chosen
    attributes name (length of this list should be equal to listOfNodeIDs)

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

    producerIDs = go.getListOfNodeIDs(myNW, actors=['producer'])
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
       listOfNodeIDs=producerIDs,
       nameOfAttr='ags',
       listOfAttr=listOfAGS
    )
    print('Is it stored into the graph: '+str(didItWork))



    ===========================================================================
    """
    done = False

    # if no list is given, assume user wants all nodes
    if listOfNodeIDs == None:
        listOfNodeIDs = list(dict(SCGraph.nodes(data=True)).keys())

    # make sure both input lists are, indeed, lists
    isNodeIDsaList = isinstance(listOfNodeIDs, list)
    isAttrssaList = isinstance(listOfAttr, list)
    if not isNodeIDsaList:
        raise Exception("Node IDs ist not a list!")
    if not isAttrssaList:
        raise Exception("Attributes ist not a list!")

    # should be of the same length!
    if len(listOfNodeIDs) == len(listOfAttr):
        done = True
    else:
        raise Exception(
            "length of list of node ids is not equal " "to length of list of attributes"
        )

    # TODO add attribute to graph and make sure the order is preserved....
    # assume both lists are ordered correctly
    nodes = zip(listOfNodeIDs, [{nameOfAttr: al} for al in listOfAttr])
    SCGraph.add_nodes_from(list(nodes))

    return done


def getNodeIDswAttr(SCGraph: Graph, nameOfAttr: str, listOfNodeIDs=None):
    """
    TODO test, check if nodes exists?
    purpose: get list of tuple with (nid,content of nodes with nid)
    if no list of nodes are given, all nodes with this attribute are returned

    :param SCGraph: the graph created with
    createGraph(listOfActors: list, listOfLocations: list,
    listOfProducts: list)
    :param nameOfAttr: name of the existing attribute
    :param listOfNodeIDs(optional): give a prepared list of node IDs

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
    if listOfNodeIDs == None:
        listOfNodeIDs = list(dict(SCGraph.nodes(data=True)).keys())

    # make sure attribute exists already!
    allAttrSoFar = getExistingAttrs(SCGraph, gtype="nodes")
    if nameOfAttr not in allAttrSoFar:
        raise Exception(
            "Attribute given is not added to " "any node in the graph, please check!"
        )

    # make sure input list is, indeed, a list
    isNodeIDsaList = isinstance(listOfNodeIDs, list)
    if not isNodeIDsaList:
        raise Exception("Node IDs ist not a list!")

    # get all nodes
    attr = nx.get_node_attributes(SCGraph, nameOfAttr).items()
    myListOfTuples = [val for val in attr if val[0] in listOfNodeIDs]

    return myListOfTuples


def convertLoT2NTs(myListOfTuples: list):
    """
    purpose: convert/unpack a list of 2-tuples to two tuple lists
    usage: x,y = convertLoT2NTs(thelistoftuples)
    for turning it back, see: convertTup2LoT

    :param myListOfTuples: a list of tuples

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
    return list(map(list, list(zip(*myListOfTuples))))


def convertTup2LoT(t1, t2):
    """
    TODO: test
    purpose: convert 2 lists or 2 tuples into 1 list of tuples
    if first argument is list of tuples, the return will be a list of 3 tuples
    usage;
    x = convertTup2LoT(tuple or list ,tuple or list)                or
    x = convertTup2LoT(list of tuples, tuple or list )
    for turning back, see convertLoT2NTs

    :param t1: tuple or list
    :param t2: tuple or list

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
    if type(t1) == list and all(isinstance(i, tuple) for i in t1):
        t1o, t1i = list(zip(*t1))
        myOutput = list(zip(t1o, t1i, t2))
    else:
        myOutput = list(zip(t1, t2))
    return myOutput


def proxyModel(inputNr: float, proxyData: list, proxyNr: float) -> list:
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

    :param inputNr: float. (usage see explanation above)
    :param proxyData: list or List of tuples
    (if you want to connect your node IDs with the respective attribute).
    (usage see explanation above)
    :param proxyNr: float. (usage see explanation above)


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
        nid, pd = convertLoT2NTs(proxyData)
    except ValueError:
        oid, iid, pd = convertLoT2NTs(proxyData)
    except TypeError:
        pd = proxyData
        print(
            "Warning: proxyModel expected a list of tuples, "
            "but does its job anyway. "
            "\nReturned a list of modelled data with no Node IDs!"
        )

    outputData = [el / proxyNr * inputNr for el in pd]
    try:
        outputData = convertTup2LoT(nid, outputData)
    except NameError:
        try:
            outputData = list(zip(oid, iid, outputData))
        except UnboundLocalError:
            pass

    return outputData


def getExistingAttrs(SCGraph: Graph, gtype="n"):
    """
    purpose: return a list of attributes that exist in this graph

    possible are nodes: 'nodes' or 'n'
    possible are edges: 'edges' or 'e'

    :param SCGraph: the graph in question
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

    if gtype == "nodes" or gtype == "n":
        listOfAttributes = list(
            set(
                chain(
                    *[list(x.keys()) for x in dict(SCGraph.nodes(data=True)).values()]
                )
            )
        )
    elif gtype == "edges" or gtype == "e":
        listOfAttributes = list(
            set(
                chain(
                    *[
                        list(x.keys())
                        for x in [w for _, _, w in list(SCGraph.edges(data=True))]
                    ]
                )
            )
        )
    else:
        print("Error: This input was confusing, please try again")
        listOfAttributes = []

    return listOfAttributes


def getAllCombinations(list1: list, list2: list, order="1st"):
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
    if order == "1st":
        allCombinations1 = [o for _ in list2 for o in list1]
        allCombinations2 = [i for i in list2 for _ in list1]
    elif order == "2nd":
        allCombinations1 = [i for i in list1 for _ in list2]
        allCombinations2 = [o for _ in list1 for o in list2]
    else:
        raise Exception("please set the parameter 'order' to either " "'1st' or '2nd'")

    return (allCombinations1, allCombinations2)


def getEdgeID(SCGraph: Graph, outgoingNodes: list, incomingNodes: list):
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
    :param outgoingNodes: a list of NodeIDs of existing nodes in SCGraph
    (can be retrieved with getNodeID)
    :param incomingNodes: a list of NodeIDs  of existing nodes in SCGraph
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
    completeListOfAllNodesinGraph = [n for n in dict(SCGraph.nodes(data=True)).keys()]

    allThere = all(
        [
            all(r in completeListOfAllNodesinGraph for r in outgoingNodes),
            all(r in completeListOfAllNodesinGraph for r in incomingNodes),
        ]
    )

    if allThere:
        listOfEdgeIDs = convertTup2LoT(outgoingNodes, incomingNodes)
    else:
        outHere = all(r in completeListOfAllNodesinGraph for r in outgoingNodes)
        inHere = all(r in completeListOfAllNodesinGraph for r in incomingNodes)
        if not outHere:
            raise Exception(
                "found some nodes that are not existing! " "(check outgoing nodes)"
            )
        if not inHere:
            raise Exception(
                "found some nodes that are not existing! " "(check incoming nodes)"
            )
    return listOfEdgeIDs


def addAttr2Edges(
    SCGraph: Graph, listOfEdgeIDs: list, listOfContent: list, attr="weight"
) -> bool:
    """
    purpose: add an attribute to a given list of edge IDs
    NOTE that if an edge ID is not recognized, a node with the address given
    will be created with no attributes!

    :param SCGraph: This is the graph in question.
    :param listOfEdgeIDs: a list of 2-tuples gotten from getEdgeID
    :param listOfContent: a list of numbers(e.g. double or int) in the same
    order as the list of edges
    :param attr(optional): default='weight'. Another str may be given
    (e.g. distance).

    :return: Boolean. True if all was entered into the graph correctly.

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

    producerIDs = go.getListOfNodeIDs(
        myNW,
        actors=['producer']
    )
    consumerOfSchnapsIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        products=['schnaps'],
        actors=['producer']
    )
    myEdgeIDs = go.getEdgeID(
        SCGraph=myNW,
        outgoingNodes=producerIDs,
        incomingNodes=consumerOfSchnapsIDs
        )

    listOfShipping = [10,1,2000]
    listOfDistances = [50.2,10.3,111.2]

    didItWork1 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdgeIDs,
        listOfContent=listOfShipping
    )
    didItWork2 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdgeIDs,
        listOfContent=listOfDistances,
        attr='distance'
    )

    print('Is the weight stored into the graph: '+ str(didItWork1))
    print('Is the distance stored into the graph: ' + str(didItWork2))

    ===========================================================================
    """
    done = False
    a, b = convertLoT2NTs(listOfEdgeIDs)
    myListofTuples = list(zip(a, b, listOfContent))

    if len(listOfEdgeIDs) != len(listOfContent):
        raise Exception(
            "Error: length of input lists are unequal("
            + str(len(listOfEdgeIDs))
            + " vs. "
            + str(len(listOfContent))
            + ")"
        )

    try:
        SCGraph.add_weighted_edges_from(myListofTuples, attr)
        done = True
    except TypeError:
        raise Exception("Something from the networkx package went wrong")
    return done


def getEdgesAttr(SCGraph: Graph, attr="weight", listOfEdgeIDs=None):
    """
    purpose: get me the edge attributes content from the edges i provide.

    :param SCGraph: the graph in question.
    :param attr(optional): An existing attribute of an edge in the graph.
    default:'weight'. Learn about existing attributes in the graph via
    the function "getExistingAttrs".
    :param listOfEdgeIDs(optional): A list of edge IDs given. default:
    all active edges (gotten from "getEdgeID").

    :return: A list of attribute content in the same order as the edge
    IDs given.

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

    producerIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer']
    )
    consumerOfSchnapsIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        products=['schnaps'],
        actors=['producer']
    )
    myEdgeIDs = go.getEdgeID(
        SCGraph=myNW,
        outgoingNodes=producerIDs,
        incomingNodes=consumerOfSchnapsIDs
    )

    listOfShipping = [10, 1, 2000]
    listOfDistances = [50.2, 10.3, 111.2]

    didItWork1 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdgeIDs,
        listOfContent=listOfShipping
    )
    didItWork2 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdgeIDs,
        listOfContent=listOfDistances,
        attr='distance'
    )

    # all active edges with attribute 'weight'
    print(go.getEdgesAttr(SCGraph=myNW))

    # active edges with attribute 'distance'
    print(
        go.getEdgesAttr(
            SCGraph=myNW,
            attr = 'distance',
            listOfEdgeIDs = [myEdgeIDs[1]]
        )
    ) # just one edge



    ===========================================================================
    """
    # if no list is given, assume user wants all nodes
    allEdgeIDs = [(o, i) for o, i, _ in SCGraph.edges(data=attr)]
    allAttrs = [w for _, _, w in SCGraph.edges(data=attr)]

    if listOfEdgeIDs == None:
        listOfEdgeIDs = allEdgeIDs
        listOfAttrs = allAttrs
    else:
        try:
            adress = [allEdgeIDs.index(e) for e in listOfEdgeIDs]
            listOfAttrs = [allAttrs[a] for a in adress]
        except ValueError:
            raise Exception("at least one edge ID is wrong!")
    return listOfAttrs


def getEdgeIDswAttr(SCGraph: Graph, attr="weight", listOfEdgeIDs=None):
    """
    purpose: get me the edge attributes content from the edges i provide,
    but include the edge ID in the list returned. Good for transparency.

    :param SCGraph: the graph in question.
    :param attr(optional): An existing attribute of an edge in the graph.
    default:'weight'. Learn about existing attributes in the graph via
    the function "getExistingAttrs".
    :param listOfEdgeIDs(optional): A list of edge IDs given. default:
    all active edges (gotten from "getEdgeID").

    :return: a list of tuples with
    (outgoing node IDs, incoming node IDs, chosen attr)
    may be converted to 3 lists with
    nodeOut, nodeIn, weight = convertLoTToNTs(
        getEdgeIDswAttr(SCGraph, attr, listofEdgeIDs)
    )

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

    producerIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['schnaps']
    )
    consumerOfSchnapsIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        products=['schnaps'],
        actors=['producer']
    )
    myEdgeIDs = go.getEdgeID(
        SCGraph=myNW,
        outgoingNodes=producerIDs,
        incomingNodes=consumerOfSchnapsIDs
    )

    listOfShipping = [10,1,2000]
    listOfDistances = [50.2,10.3,111.2]

    didItWork1 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdgeIDs,
        listOfContent=listOfShipping
    )
    didItWork2 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdgeIDs,
        listOfContent=listOfDistances,
        attr='distance'
    )

    print(go.getEdgeIDswAttr(SCGraph=myNW, attr = 'weight'))
    print(go.getEdgeIDswAttr(
            SCGraph=myNW,
            attr='distance',
            listOfEdgeIDs=myEdgeIDs
        )
    )

    nodeOut, nodeIn, weight = go.convertLoT2NTs(
        go.getEdgeIDswAttr(
            SCGraph=myNW,
            attr='distance',
            listOfEdgeIDs=[myEdgeIDs[1]]
        )
    )
    edgeID = go.convertTup2LoT(nodeOut,nodeIn)



    ===========================================================================
    """
    # if no list is given, assume user wants all nodes
    allEdgeIDs = [(o, i) for o, i, _ in SCGraph.edges(data=attr)]
    allEwA = [(o, i, w) for o, i, w in SCGraph.edges(data=attr)]

    if listOfEdgeIDs == None:
        listOfEdgeIDs = allEdgeIDs
        listofBoth = allEwA
    else:
        try:
            adress = [allEdgeIDs.index(e) for e in listOfEdgeIDs]
            listofBoth = [allEwA[a] for a in adress]
        except ValueError:
            raise Exception("at least one edge ID is wrong!")
    return listofBoth


def minOptFlowForBeta(
    SCGraph: Graph,
    listOfSenderIDs: list,
    listOfReceiverIDs: list,
    beta: float,
    edgeattrDistName: str,
    distanceModel: str,
):
    """
    purpose: minimization of the functional for optimal flow
    for hyman to work we need a range of minimum average transport distance
    this calculates the optimum flow by minimizing
    sum_{i,j}(flow_{i,j}*dist_{i,j})

    :param SCGraph: the graph on which this optimal flow is calculated
    :param listOfSenderIDs: the list of node IDs which supply the wares
    :param listOfReceiverIDs: the list of node IDs which demand the wares
    :param beta: variable for minimization.
    free parameter (float) determining the average amount of flow.
    (see hymanModel for details)
    :param edgeattrDistName: attribute in the graph which contains the distance
    between the nodes
    NOTE that this information needs to be stored on the edges between the
    participating nodes. Otherwise this function will fail.
    :param distanceModel: the distance model for the hymanModel
    (see hymanModel for details)

    :return: variable beta for optimimum flow. needed for minimum transport
    distance.

    example:
    internal function. needed for calcPotTransportDistances. Not part of the
    user experience.



    ===========================================================================
    """
    return float(
        optimize.minimize(
            optFlow,
            beta,
            args=(
                SCGraph,
                listOfSenderIDs,
                listOfReceiverIDs,
                edgeattrDistName,
                distanceModel,
            ),
        ).x
    )


def optFlow(
    beta: float,
    SCGraph: Graph,
    listOfSenderIDs: list,
    listOfReceiverIDs: list,
    edgeattrDistName: str,
    distanceModel: str,
):
    """
    purpose: calculate the flow between the senders and the receivers of goods
    given the free parameter beta, the distances between all locations of
    participants and the distance model

    :param beta: free parameter (float) determining the average amount of flow
    :param SCGraph: the graph with information about outgoing goods from
    senders and incoming goods for receivers
    :param listOfSenderIDs: list of node IDs in G (should have 'output'
    attribute)
    :param listOfReceiverIDs: list of node IDs in G (should have 'input'
    attribute)
    :param edgeattrDistName: str name of the edge attribute for the
    geographical distance between nodes
    listOfSenderIDs, order of rows should be identical to listOfReceiverIDs,
    NOTE hymanModel expects this to be 'distance' !
    :param distanceModel: 'exp' for exponential, 'lin' for linear

    :return: flow between listOfSenderIDs and listOfReceiverIDs

    example:
    internal function. needed for calcPotTransportDistances. Not part of the
    user experience.



    ===========================================================================
    """
    sendID, sendOut = convertLoT2NTs(
        getNodeIDswAttr(
            SCGraph=SCGraph, nameOfAttr="output", listOfNodeIDs=listOfSenderIDs
        )
    )
    recID, recIn = convertLoT2NTs(
        getNodeIDswAttr(
            SCGraph=SCGraph, nameOfAttr="input", listOfNodeIDs=listOfReceiverIDs
        )
    )

    if sendID == []:
        raise Exception(
            "please enter attribute OUTPUT with " "corresponding content to the network"
        )
    if recID == []:
        raise Exception(
            "please enter attribute INPUT with " "corresponding content to the network"
        )

    distanceMatrix = getDistMatrix(
        SCGraph, listOfSenderIDs, listOfReceiverIDs, edgeattrDistName
    )

    if distanceModel == "exp":
        dM = np.exp(-beta * distanceMatrix)
    if distanceModel == "lin":
        dM = beta * distanceMatrix

    aggregates = [sendOut, recIn]
    dimensions = [[0], [1]]
    # initial furness starts!
    try:
        IPF = ipfn(dM, aggregates, dimensions)  # (linux based systems)
    except TypeError:
        IPF = ipfn.ipfn(dM, aggregates, dimensions)  # (windows based systems)

    dM = IPF.iteration()

    flowMatrix = np.sum(dM * distanceMatrix)
    return flowMatrix


def getDistMatrix(
    SCGraph: Graph,
    listOfSenderIDs: list,
    listOfReceiverIDs: list,
    edgeattrDistName: str,
):
    """
    purpose: If all edges between listOfSenderIDs and listOfReceiverIDs
    contain the attribute edgeattrDistName, this function returns a numpy
    matrix of all distances between all combinations of participating nodes.
    Needed for hymanModel, but user may need it for other purposes.
    Technically, this can be used as a getEdgeAttribute function.

    :param SCGraph: the graph with information about outgoing goods from
    senders and incoming goods for receivers
    :param listOfSenderIDs: list of node IDs in G (should have 'output'
    attribute)
    :param listOfReceiverIDs: list of node IDs in G (should have 'input'
    attribute)
    :param edgeattrDistName: str name of the edge attribute for the
    geographical distance between nodes listOfSenderIDs, order of rows should
    be identical to listOfReceiverIDs,
    NOTE hymanModel expects this to be 'distance' !

    :return: a numpy matrix of all combinations of distances between
    participating nodes

    example:
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk','beer','schnaps']
    act = ['producer','consumer','warehouse','store']
    loc = ['BER','SXF','TXL']
    myNW = go.createGraph(
        listOfActors=act,
        listOfLocations=loc,
        listOfProducts=prod
    )

    # Creating Distance matrix
    myData={loc[0]:[1,2,50],loc[1]:[2,1,49],loc[2]:[50,49,1]}
    myDF = pd.DataFrame(myData,index=loc)

    # creating list of node IDs of participants
    senderIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['milk']
    )
    receiverIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['consumer'],
        products=['milk']
    )

    # creating edgeIDs for adding 'distance' to graph
    allcombIDs1,allcombIDs2 = go.getAllCombinations(
        list1=senderIDs,
        list2=receiverIDs,
        order='1st'
    )
    myEdges4Graph = go.getEdgeID(
        SCGraph=myNW,
        outgoingNodes=allcombIDs1,
        incomingNodes=allcombIDs2
    )
    didItWork3 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdges4Graph,
        listOfContent=myDF.values.flatten(),
        attr='distance'
    )

    print(
        go.getDistMatrix(
            SCGraph=myNW,
            listOfSenderIDs=senderIDs,
            listOfReceiverIDs=receiverIDs,
            edgeattrDistName='distance'
        )
    )




    ===========================================================================
    """
    po, pi = getAllCombinations(listOfSenderIDs, listOfReceiverIDs)
    myEdges = getEdgeID(SCGraph, po, pi)
    myLoD = getEdgesAttr(SCGraph, attr=edgeattrDistName, listOfEdgeIDs=myEdges)
    distanceMatrix = np.reshape(myLoD, (len(listOfSenderIDs), len(listOfReceiverIDs)))
    return distanceMatrix


def propFlow(SCGraph: Graph, listOfSenderIDs: list, listOfReceiverIDs: list):
    """
    purpose: calculate the flow between the senders and the receivers of goods
    given a proportional flow of goods.
    flow_{i,j} = out_i * in_j / (sum_i out_i) = out_i * in_j / (sum_j in_j)
    Needed for calcPotTransportDistances.

    :param SCGraph: the graph with information about outgoing goods from
    senders and incoming goods for receivers
    :param listOfSenderIDs: list of node IDs in G (should have 'output'
    attribute)
    :param listOfReceiverIDs: list of node IDs in G (should have 'input'
    attribute)

    :return: flow between listOfSenderIDs and listOfReceiverIDs

    example:
    internal function. needed for calcPotTransportDistances. Not part of the
    user experience.



    ===========================================================================
    """
    sendID, sendOut = convertLoT2NTs(
        getNodeIDswAttr(
            SCGraph=SCGraph, nameOfAttr="output", listOfNodeIDs=listOfSenderIDs
        )
    )
    recID, recIn = convertLoT2NTs(
        getNodeIDswAttr(
            SCGraph=SCGraph, nameOfAttr="input", listOfNodeIDs=listOfReceiverIDs
        )
    )

    if sendID == []:
        raise Exception(
            "please enter attribute OUTPUT with " "corresponding content to the network"
        )
    if recID == []:
        raise Exception(
            "please enter attribute INPUT with " "corresponding content to the network"
        )

    flow = np.outer(
        np.array(sendOut, dtype=np.int64), np.array(recIn, dtype=np.int64)
    ) / np.sum(np.array(sendOut, dtype=np.int64))
    return flow


def getWeightedDist(flow, dist) -> float:
    """
    purpose: calculates sum_{i,j} flow_{i,j}*dist_{i,j} / sum_{i,j} flow_{i,j}.
    Needed for calculating optimal flow, gravity model and transport distance.

    :param flow: np array as the result of furnessModel
    :param dist: np array as the distances of locations involved

    :return: (float). The weighted Distance based on the flow between sender and receiver
    and their respective distance.

    example:
    internal function. needed for calcPotTransportDistances. Not part of the user experience.



    ===============================================================================================
    """
    WeightDist = np.sum(flow * dist) / (np.sum(flow))
    return WeightDist


def calcPotTransportDistances(
    SCGraph: Graph, listOfSenderIDs: list, listOfReceiverIDs: list, nrOfValues=5
) -> list:
    """
    purpose: calculates nrOfValues possible transport distances, given a range
    of possible values. The minimum value of this range is the optimum flow
    (see optFlow for details).
    The maximum value of this range is the proportional flow (see propFlow for
    details).
    Both values are calculated internally and depend on the flow between
    participants of supply and demand regarding a specific product.
    NOTE that the node attributes 'output' for the supplier nodes and 'input'
    for the receiver nodes of goods, as well as the edge attributes 'distance'
    and 'weight' between ALL the participating nodes MUST NOT be missing!

    :param SCGraph: the graph with information about outgoing goods from
    senders and incoming goods for receivers
    :param listOfSenderIDs: list of node IDs in G (should have 'output'
    attribute)
    :param listOfReceiverIDs: list of node IDs in G (should have 'input'
    attribute)
    :param nrOfValues: (optional) int value. Please insert the number of
    possible transport distances (default is 5).

    :return: a list of possible transport distances. needed for hymanModel to
    determine beta (details see hymanModel)

    example:
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk','beer','schnaps']
    act = ['producer','consumer','warehouse','store']
    loc = ['BER','SXF','TXL']
    myNW = go.createGraph(
        listOfActors=act,
        listOfLocations=loc,
        listOfProducts=prod
    )

    # creating list of node IDs of participants
    senderIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['milk']
    )
    receiverIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['consumer'],
        products=['milk']
    )


    # Creating Distance matrix
    myData={loc[0]:[1,2,50],loc[1]:[2,1,49],loc[2]:[50,49,1]}
    myDF = pd.DataFrame(myData,index=loc)

    # creating edgeIDs for adding 'distance' to graph
    allcombIDs1,allcombIDs2 = go.getAllCombinations(
        senderIDs,
        receiverIDs,
        order='1st'
    )
    myEdges4Graph = go.getEdgeID(myNW,allcombIDs1,allcombIDs2)
    didItWork3 = go.addAttr2Edges(
        myNW,myEdges4Graph,
        myDF.values.flatten(),
        attr='distance'
    )


    # supply and demand
    sendingMilk = [10,50,40]
    receivingMilk = [30,30,40]
    go.addAttr2ExistingNodes(myNW,senderIDs,'output',sendingMilk)
    go.addAttr2ExistingNodes(myNW,receiverIDs,'input',receivingMilk)

    myTDs = go.calcPotTransportDistances(
        SCGraph=myNW,
        listOfSenderIDs=senderIDs,
        listOfReceiverIDs=receiverIDs
    )



    ===========================================================================
    """
    # should not depend on beta, but optimization needs a starting value,
    # thus hardcoded here
    betaHere = 0.1
    edgeattrDistName = "distance"
    distanceModel = "exp"

    distanceMatrix = getDistMatrix(
        SCGraph, listOfSenderIDs, listOfReceiverIDs, edgeattrDistName
    )
    betaOpt = minOptFlowForBeta(
        SCGraph,
        listOfSenderIDs,
        listOfReceiverIDs,
        betaHere,
        edgeattrDistName,
        distanceModel,
    )
    flowopt = furnessModel(
        SCGraph,
        listOfSenderIDs,
        listOfReceiverIDs,
        betaOpt,
        edgeattrDistName,
        distanceModel,
    )

    # min dist is  optimum flow distance
    minDist = getWeightedDist(flowopt, distanceMatrix)

    # max dist is proportional flow distance
    flowprop = propFlow(SCGraph, listOfSenderIDs, listOfReceiverIDs)
    maxDist = getWeightedDist(flowprop, distanceMatrix)

    # return the best ensemble members of transport distances
    return list(
        np.round(np.linspace(minDist, maxDist, nrOfValues + 2)[1 : nrOfValues + 1])
    )


def hymanModel(
    SCGraph: Graph,
    listOfSenderIDs: list,
    listOfReceiverIDs: list,
    transportDistance: float,
    tolerance=0.1,
):
    """
    purpose: Gravity Model. calculate the flow of goods, based on a transport
    distance (got from calcPotTransportDistances), by estimating unkown beta.
    Hardcoded here is the exponential distance model, input for furnessModel
    (see furnessModel for details).
    If the user wants to create their own distance model, the user may start
    with furness, probably wants to create their own version of hyman, as well.

    1. initial guess of beta0 = 1./transportDistance
    2. calculate flow based on beta_0: flow_0=furnessModel(beta_0)
    3. calculate currentTransportDistance_0 =
        getWeightedDist(flow_0,getDistMatrix)
    4. start iterative process
        4a. calculate beta_i(beta_i-1, currentTransportDistance_0)
        4b. calculate flow_i=furnessModel(beta_i)
        4c. calculate currentTransportDistance_i =
            getWeightedDist(flow_i,getDistMatrix)
        4d. check if abs(currentTransportDistance_i - transportDistance)
            < tolerance -> break cycle
    5. return flow=furnessModel(beta_i)

    :param SCGraph: the graph with information about outgoing goods from
    senders and incoming goods for receivers
    :param listOfSenderIDs: list of node IDs in G (should have 'output'
    attribute)
    :param listOfReceiverIDs: list of node IDs in G (should have 'input'
    attribute)
    :param transportDistance: weighted average transport distance. Calculated
    via calcPotTransportDistances
    :param tolerance: (optional) parameter of how good the model should fit
    the given input data. default is 0.01 of the unit 'weight' of the edges
    of the graph

    :return: flow between supplier and receiver nodes based on given transport
    distance. Needed for calibration vs. total flow of all goods

    example:
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk','beer','schnaps']
    act = ['producer','consumer','warehouse','store']
    loc = ['BER','SXF','TXL']
    myNW = go.createGraph(
        listOfActors=act,
        listOfLocations=loc,
        listOfProducts=prod
    )


    # creating list of node IDs of participants
    senderIDs = go.getListOfNodeIDs(
        myNW,
        actors=['producer'],
        products=['milk']
    )
    receiverIDs = go.getListOfNodeIDs(
        myNW,
        actors=['consumer'],
        products=['milk']
    )


    # Creating Distance matrix
    myData={loc[0]:[1,2,50],loc[1]:[2,1,49],loc[2]:[50,49,1]}
    myDF = pd.DataFrame(myData, index=loc)

    # creating edgeIDs for adding 'distance' to graph
    allcombIDs1,allcombIDs2 = go.getAllCombinations(
        senderIDs,
        receiverIDs,
        order='1st'
    )
    myEdges4Graph = go.getEdgeID(myNW, allcombIDs1, allcombIDs2)
    didItWork3 = go.addAttr2Edges(
        myNW,
        myEdges4Graph,
        myDF.values.flatten(),
        attr='distance'
    )


    # supply and demand
    sendingMilk = [10,50,40]
    receivingMilk = [30,30,40]
    go.addAttr2ExistingNodes(myNW,senderIDs,'output',sendingMilk)
    go.addAttr2ExistingNodes(myNW,receiverIDs,'input',receivingMilk)

    # get the transport distances
    myTDs = go.calcPotTransportDistances(
        myNW,
        listOfSenderIDs=senderIDs,
        listOfReceiverIDs=receiverIDs
    )

    # Run the gravity model with given transport distance and return the flow
    print(
        go.hymanModel(
            myNW,
            listOfSenderIDs=senderIDs,
            listOfReceiverIDs=receiverIDs,
            transportDistance=myTDs[0],
            tolerance = 0.01
        )
    )



    ===========================================================================
    """
    betaList = []
    distList = []
    countLoops = 0
    edgesAttr = "distance"
    distModel = "exp"

    # initialising hyman
    beta0 = 1.0 / transportDistance
    betaList.append(beta0)
    flow0 = furnessModel(
        SCGraph, listOfSenderIDs, listOfReceiverIDs, beta0, edgesAttr, distModel
    )

    distanceMatrix = getDistMatrix(
        SCGraph, listOfSenderIDs, listOfReceiverIDs, edgesAttr
    )
    dist0 = getWeightedDist(flow0, distanceMatrix)
    distList.append(dist0)

    # starting the potential infinite loop
    # TODO sanity check!
    if abs(transportDistance - distList[countLoops]) <= tolerance:
        flow = flow0
    while abs(transportDistance - distList[countLoops]) > tolerance:
        if countLoops == 0:
            beta1 = beta0 * distList[countLoops] / transportDistance
            betaList.append(beta1)
        elif countLoops > 0:
            betaNext = np.abs(
                (
                    (transportDistance - distList[countLoops - 1])
                    * betaList[countLoops]
                    - (transportDistance - distList[countLoops])
                    * betaList[countLoops - 1]
                )
                / (distList[countLoops] - distList[countLoops - 1])
            )
            betaList.append(betaNext)
        betaThisTime = betaList[countLoops + 1]

        flow = furnessModel(
            SCGraph,
            listOfSenderIDs,
            listOfReceiverIDs,
            betaThisTime,
            edgesAttr,
            distModel,
        )
        distThisTime = getWeightedDist(flow, distanceMatrix)
        distList.append(distThisTime)

        countLoops += 1

        # break if in local minimum and check if any dist was closer
        # to the given transport distance
        if countLoops > 20:
            if abs(distList[countLoops] - distList[countLoops - 5]) < 0.001:
                betaBest = betaList[distList.index(min(distList))]
                flow = furnessModel(
                    SCGraph,
                    listOfSenderIDs,
                    listOfReceiverIDs,
                    betaBest,
                    edgesAttr,
                    distModel,
                )
                break

        # break if minimization routine explodes due to numerical issues
        if betaThisTime > 50:
            betaBest = betaList[distList.index(min(distList))]
            flow = furnessModel(
                SCGraph,
                listOfSenderIDs,
                listOfReceiverIDs,
                betaBest,
                edgesAttr,
                distModel,
            )
            break

        print(
            "On the %2d. iteration: tolerance is down to %3.4f"
            % (countLoops, abs(transportDistance - distList[countLoops]))
        )

        if np.isnan(transportDistance):
            raise Exception(
                "something went wrong, " "given transport distance returned nan!"
            )
        if np.isnan(distList[countLoops]):
            raise Exception(
                "something went wrong, " "current transport distance returned nan!"
            )

    # check if by running into a local minimum,
    # a transport distance was found that was better than the last one
    tolThisTime = np.abs(transportDistance - distThisTime)
    tolBest = np.abs([transportDistance - d for d in distList]).tolist()
    if tolThisTime > tolBest[tolBest.index(min(tolBest))]:
        betaBest = betaList[tolBest.index(min(tolBest))]
        flow = furnessModel(
            SCGraph, istOfSenderIDs, listOfReceiverIDs, betaBest, edgesAttr, distModel
        )
    print(
        "On the last iteration (%2d.): tolerance is down to %3.4f"
        % (tolBest.index(min(tolBest)), tolBest[tolBest.index(min(tolBest))])
    )
    return flow


def furnessModel(
    SCGraph: Graph,
    listOfSenderIDs: list,
    listOfReceiverIDs: list,
    beta: float,
    edgeattrDistName="distance",
    distanceModel="exp",
):
    """
    purpose: calculate the flow between the senders and the receivers of goods
    given the free parameter beta, the distances between all locations of
    participants and the distance model.

    Heart of the gravity model with exponential distance model.
    flow_i,j = N_i,j * O_i * D_j * exp(-beta*distmatrix_i,j)

    - N_i,j is normalization values, determined by furness automatically.
    - O_i is the goods sent out by supplier i.
    - D_j is the goods demanded by receiver j.
    - beta is a free parameter determining the average amount of flow
    (determined by hymanModel)
    - distmatrix_i,j is the geographical distance between supplier i and
    receiver j (got from getDistMatrix)


    :param SCGraph: the graph with information about outgoing goods from
    senders and incoming goods for receivers
    :param listOfSenderIDs: list of node IDs in G (should have 'output'
    attribute)
    :param listOfReceiverIDs: list of node IDs in G (should have 'input'
    attribute)
    :param beta: free parameter (float) determining the average amount of flow
    :param edgeattrDistName(optional): str name of the edge attribute for the
    geographical distance between nodes listOfSenderIDs, order of rows should
    be identical to listOfReceiverIDs. Default is 'distance'
    :param distanceModel(optional): 'exp'(default) for exponential, 'lin' for
    linear

    :return: flow between supplier and receiver nodes based on given beta.

    example:
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk','beer','schnaps']
    act = ['producer','consumer','warehouse','store']
    loc = ['BER','SXF','TXL']
    myNW = go.createGraph(
        listOfActors=act,
        listOfLocations=loc,
        listOfProducts=prod
    )

    # creating list of node IDs of participants
    senderIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['milk']
    )
    receiverIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['consumer'],
        products=['milk']
    )

    # Creating Distance matrix
    myData={loc[0]:[1,2,50],loc[1]:[2,1,49],loc[2]:[50,49,1]}
    myDF = pd.DataFrame(myData,index=loc)

    # creating edgeIDs for adding 'distance' to graph
    allcombIDs1,allcombIDs2 = go.getAllCombinations(
        senderIDs,
        receiverIDs,
        order='1st'
    )
    myEdges4Graph = go.getEdgeID(myNW,allcombIDs1,allcombIDs2)
    didItWork3 = go.addAttr2Edges(
        myNW,
        myEdges4Graph,
        myDF.values.flatten(),
        attr='distance'
    )

    # supply and demand
    sendingMilk = [10,50,40]
    receivingMilk = [30,30,40]
    go.addAttr2ExistingNodes(myNW,senderIDs,'output',sendingMilk)
    go.addAttr2ExistingNodes(myNW,receiverIDs,'input',receivingMilk)

    # determining beta
    myBeta = 0.1
    flow = go.furnessModel(
        myNW,
        senderIDs,
        receiverIDs,
        myBeta,
        'distance',
        'exp'
    )



    ===========================================================================
    """
    sendID, sendOut = convertLoT2NTs(
        getNodeIDswAttr(
            SCGraph=SCGraph, nameOfAttr="output", listOfNodeIDs=listOfSenderIDs
        )
    )
    recID, recIn = convertLoT2NTs(
        getNodeIDswAttr(
            SCGraph=SCGraph, nameOfAttr="input", listOfNodeIDs=listOfReceiverIDs
        )
    )

    if sendID == []:
        raise Exception(
            "please enter attribute OUTPUT with " "corresponding content to the network"
        )
    if recID == []:
        raise Exception(
            "please enter attribute INPUT with " "corresponding content to the network"
        )

    distanceMatrix = getDistMatrix(
        SCGraph, listOfSenderIDs, listOfReceiverIDs, edgeattrDistName
    )

    if distanceModel == "exp":
        dM = np.exp(-beta * distanceMatrix)
    if distanceModel == "lin":
        dM = beta * distanceMatrix

    aggregates = [sendOut, recIn]
    dimensions = [[0], [1]]
    # initial furness starts!
    try:
        IPF = ipfn(dM, aggregates, dimensions)  # (linux based systems)
    except TypeError:
        IPF = ipfn.ipfn(dM, aggregates, dimensions)  # (windows based systems)

    dM = IPF.iteration()
    # initial furness is finished
    flowMatrix = dM
    return flowMatrix


# def calibration process(nodes involved..) ->edges2network
def calibrationOLD(
    flowTensor: np.ndarray, calibrationMatrix: np.array, nrIter=100_000, wait=1000
):
    """
    purpose: [DEPRECATED] calculate the flow of goods, based on a transport
    distance (got from calcPotTransportDistances), by estimating unkown beta.
    Hardcoded here is the exponential distance model, input for furnessModel
    (see furnessModel for details).
    The hymanModel allows to calculate the flow if the average transport
    distance is known. Unfortunately only a range of possible average
    transport distances for each product are known (and may be calculated by
    the
    function calcPotTransportDistances).
    In Germany a Federal Transport Infrastructure Plan(FTIP) exists, that gives
    information about how much food is transported in average from
    one county to another.
    For this to be possible to be calibrated against the flow of all
    products involved needs to summed up.
    calcPotTransportDistances returns for every product a number of possible
    average transport distances (m).

    Manual: create a 4D-numpy array
    >> import numpy as np
    >> np.ndarray(
        (rowsOfFlow, colsOfFlow, nrOfProducts, nrOfTransportationDistances)
    )
    with
    - rowsOfFlow, colsOfFlow defined by the dimensions of your flow matrix
    - nrOfProducts as the number of products in your supply chain
    - nrOfTransportationDistances as the number of transport distances from
    calcPotTransportDistances
    This creates one of the inputs for this function: flowTensor
    The other is the calibration measure (expected is of dimension
    [rowsOfFlow,colsOfFlow]).
    If you have a calibration measure differing from your flows, consider
    using convertMatrix from this toolkit to convert your flow to the dimension
    of calibrationMatrix.



    If for all products and all available transport distance(m) a
    flow_product_m is calculated via hymanModel, then this function chooses
    randomly an ensemble of all flow_product_m, then tries to minimize the
    difference to FTIP by switching out flow_product_<m> following a
    metropolis MC-algorithm scheme.

    The formula involved
    Min(sum_products abs(flow_products_[m] - FTIP))

    If sum_products abs(flow_products_[m_NEW] - FTIP) =<
                sum_products abs(flow_products_[m_OLD] - FTIP)
        Then m_Current = m_NEW
    Else If sum_products abs(flow_products_[m_NEW] - FTIP) >
                sum_products abs(flow_products_[m_OLD] - FTIP)
        prob = (-(exp(sum_products abs(flow_products_[m_NEW] - FTIP) >
                    sum_products abs(flow_products_[m_OLD] - FTIP))))
        If prob < random_uniform(0,1)
            m_Current = m_NEW
        Else
            m_Current = m_OLD


    :param flowTensor: 4D tensor numpy.ndarray [i,j,p,m] =
    [output,input,product,member of the ensemble]
    :param calibrationMatrix: calibration measure
    :param nrIter(optional): maximum number of iterations (default 100,000).
    Note that, the algorithm ends the minimization routine if for wait
    iterations nothing has changed.
    :param wait(optional): number of iterations waiting for something to change
    (default: 1,000). Kicks in only after 10% of maximum iterations are done.

    :return listOfBestEnsembleConfiguration: a list of ids that correspond to
    the transport distances m. For each product a specific m was chosen,
    so that the difference between the input flows and the calibration
    measure in minimized.


    example:
    import numpy as np
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk','beer','schnaps']
    act = ['producer','consumer','warehouse','store']
    loc = ['BER','SXF','TXL']
    myNW = go.createGraph(
        listOfActors=act,
        listOfLocations=loc,
        listOfProducts=prod
    )

    # Creating Distance matrix
    myData={loc[0]:[1,2,50],loc[1]:[2,1,49],loc[2]:[50,49,1]}
    myDF = pd.DataFrame(myData,index=loc)

    # creating edgeIDs for adding 'distance' to graph
    allcombIDs1,allcombIDs2 = go.getAllCombinations(
        senderIDs,
        receiverIDs,
        order='1st'
    )
    myEdges4Graph = go.getEdgeID(myNW,allcombIDs1,allcombIDs2)
    didItWork3 = go.addAttr2Edges(
        myNW,
        myEdges4Graph,
        myDF.values.flatten(),
        attr='distance'
    )

    # supply and demand
    sendingProd = [[10,50,40],[11,45,44],[15,55,30]]
    receivingProd = [[30,30,40],[40,30,30],[10,70,20]]

    myTDs = []
    allFlows = np.ndarray((3,3,3,5))

    # creating list of node IDs of participants
    for run,p in enumerate(prod):
        senderIDs = go.getListOfNodeIDs(
            SCGraph=myNW,
            actors=['producer'],
            products=[p]
            )
        receiverIDs = go.getListOfNodeIDs(
            SCGraph=myNW,
            actors=['consumer'],
            products=[p]
        )

        go.addAttr2ExistingNodes(myNW,senderIDs,'output',sendingProd[run])
        go.addAttr2ExistingNodes(myNW,receiverIDs,'input',receivingProd[run])

        # get the transport distances
        myTDs.append(go.calcPotTransportDistances(
            myNW,
            listOfSenderIDs=senderIDs,
            listOfReceiverIDs=receiverIDs)
        )

        # Run the gravity model with given transport distance and return the flow
        for it,td in enumerate(myTDs[run]):
            allFlows[
                0:len(senderIDs),
                0:len(receiverIDs),
                run,
                it
            ] = go.hymanModel(
                myNW,
                listOfSenderIDs=senderIDs,
                listOfReceiverIDs=receiverIDs,
                transportDistance=td,
                tolerance = 0.01
            )





    ===============================================================================================
    """
    outputFlow, inputFlow, products, members = flowTensor.shape
    outputCal, inputCal = np.shape(calibrationMatrix)
    if outputFlow != outputCal or inputFlow != inputCal:
        raise Exception(
            "Error: Expected dimensions of the flow "
            "matrices and the calibration measure to be "
            "identical. Please consider using the function "
            "'convertMatrix' from within this library."
        )

    # random member id chosen
    initialGuess = np.random.randint(0, members, products)
    functionalOLD = 0.0

    # initalising the sum of all flows to compare to calibration matrix
    sumFlow = np.zeros((outputFlow, inputFlow))
    for p in range(products):
        oneFlow = flowTensor[
            0:outputFlow, 0:inputFlow, p, initialGuess[p]
        ]  # choose 1 member from ensemble
        sumFlow = np.add(sumFlow, oneFlow)

    # Metropolis algorithm
    oldGuess = initialGuess
    for run in range(0, nrIter):
        functionalOLD = np.abs(sumFlow - calibrationMatrix).sum(1).sum(0)
        relativeTolerance = functionalOLD / np.sum(calibrationMatrix)
        # waiting time
        if not np.mod(run, wait):
            if run > nrIter / 10:
                if np.abs(oldTolerance - relativeTolerance) / relativeTolerance < 0.001:
                    print("nothing changed for quite a while, abort")
                    break
            oldTolerance = relativeTolerance
            print(
                "Tolerance is now: "
                + str(relativeTolerance)
                + ", iteration: "
                + str(run)
            )
        if relativeTolerance < np.finfo(float).eps / outputCal / inputCal:
            newGuess = initialGuess
            break
        else:
            # changing the ensemble product by product
            whichP = np.mod(run, products)
            oldFlow = flowTensor[0:outputFlow, 0:inputFlow, whichP, oldGuess[whichP]]

            # choose different member
            newGuess = oldGuess
            newGuess[whichP] = np.random.choice(
                [i for i in range(0, members) if not i == newGuess[0]]
            )

            # replace old flow with new flow
            newFlow = flowTensor[0:outputFlow, 0:inputFlow, whichP, newGuess[whichP]]
            newSumFlow = sumFlow - oldFlow + newFlow

            # calculate functional to minimize again
            functionalNEW = np.abs(newSumFlow - calibrationMatrix).sum(1).sum(0)

            # compare if new configurations fits better
            if functionalNEW <= functionalOLD:
                # replace old member id with new one
                oldGuess[whichP] = newGuess[whichP]
                sumFlow = newSumFlow
            else:  # if not, replace with certain probability
                # a number between 0 and 1, getting smaller, the bigger f2 is comparaed to f1
                myProb = np.exp(-(functionalNEW - functionalOLD))
                if np.random.rand() < myProb:
                    oldGuess[whichP] = newGuess[whichP]
                    sumFlow = newSumFlow
                else:
                    newGuess[whichP] = oldGuess[whichP]
                    newSumFlow = 0.0

    return newGuess


def calibration(flowTensor: np.ndarray, calibrationMatrix: np.array):
    """
    purpose: calibrate the calculated flow against a calibration measure
    calculate the flow of goods, based on a transport distance
    (got from calcPotTransportDistances), by estimating the free and
    unknown parameter beta (see furnessModel).
    Hardcoded here is the exponential distance model, input for furnessModel
    (see furnessModel for details).
    The hymanModel allows to calculate the flow if the average transport
    distance is known. Unfortunately only a range of possible average
    transport distances for each product are known (and may be calculated by
    the function calcPotTransportDistances).
    In Germany a Federal Transport Infrastructure Plan(FTIP) exists, that gives
    information about how much food is transported in average from
    one county to another.
    For this to be calibrated against the flow of all products involved needs
    to summed up. calcPotTransportDistances returns for every product a number
    of possible average transport distances (m).

    Manual for input: create a 4D-numpy array for the input parameter
    flowTensor

    >> import numpy as np
    >> np.ndarray((
        rowsOfFlow,
        colsOfFlow,
        nrOfProducts,
        nrOfTransportationDistances
    ))

    with
    - rowsOfFlow, colsOfFlow defined by the dimensions of your flow matrix
    - nrOfProducts as the number of products in your supply chain
    - nrOfTransportationDistances as the number of transport distances from
    calcPotTransportDistances

    The other is the calibration measure (expected is of dimension
    [rowsOfFlow,colsOfFlow]).
    If you have a calibration measure differing from your flows, consider using
    convertMatrix from this toolkit to convert your flow to the dimension of
    the parameter calibrationMatrix.

    The formula involved in this algorithm to find the best possible
    transport distance (m):
    Min(abs(sum_products abs(flow_products_[m] - FTIP)))


    :param flowTensor: 4D tensor numpy.ndarray [i,j,p,m] = [output, input,
    product, member of the ensemble]
    :param calibrationMatrix: calibration measure of dimension [output, input]

    :return listOfBestEnsembleConfiguration: a list of ids that correspond to
    the transport distances m. For each product a specific m was chosen,
    so that the difference between the input flows and the calibration
    measure in minimized.
    May be accessed via the command
    >> flowTensor[0:output,0:input,0:products,listOfBestEnsembleConfiguration]



    example:
    import numpy as np
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk','beer','schnaps']
    act = ['producer','consumer','warehouse','store']
    loc = ['BER','SXF','TXL']
    myNW = go.createGraph(
        listOfActors=act,
        listOfLocations=loc,
        listOfProducts=prod)

    # Creating Distance matrix
    myData={loc[0]:[1,2,50],loc[1]:[2,1,49],loc[2]:[50,49,1]}
    myDF = pd.DataFrame(myData,index=loc)

    # supply and demand
    sendingProd = [[10,50,40],[11,45,44],[15,55,30]]
    receivingProd = [[30,30,40],[40,30,30],[10,70,20]]

    myTDs = []
    allFlows = np.ndarray((3,3,3,5))

    # creating list of node IDs of participants
    for run,p in enumerate(prod):
        senderIDs = go.getListOfNodeIDs(
            myNW,
            actors=['producer'],
            products=[p]
        )
        receiverIDs = go.getListOfNodeIDs(
            myNW,
            actors=['consumer'],
            products=[p]
        )

        # creating edgeIDs for adding 'distance' to graph
        allcombIDs1, allcombIDs2 = go.getAllCombinations(
            senderIDs,
            receiverIDs,
            order='1st'
        )
        myEdges4Graph = go.getEdgeID(myNW, allcombIDs1, allcombIDs2)
        didItWork3 = go.addAttr2Edges(
            myNW,
            myEdges4Graph,
            myDF.values.flatten(),
            attr='distance'
        )

        go.addAttr2ExistingNodes(myNW,senderIDs,'output',sendingProd[run])
        go.addAttr2ExistingNodes(myNW,receiverIDs,'input',receivingProd[run])

        # get the transport distances
        myTDs.append(
            go.calcPotTransportDistances(
                myNW,
                listOfSenderIDs=senderIDs,
                listOfReceiverIDs=receiverIDs
            )
        )

        # Run the gravity model with given transport distance and return the
        # flow each myTDs[run] is a list of 5 transport distances
        # allFlows is input for the calibration algorithm
        for it, td in enumerate(myTDs[run]):
            allFlows[
                0:len(senderIDs),
                0:len(receiverIDs),
                run,
                it
            ] = go.hymanModel(
                    myNW,
                    listOfSenderIDs=senderIDs,
                    listOfReceiverIDs=receiverIDs,
                    transportDistance=td,
                    tolerance=0.01
            )

    #creating calibration measure that halfway might make sense
    allFoodTransported = sum(sum(sendingProd,[]))
    dummy = np.random.rand(8)*allFoodTransported/80+\
            allFoodTransported/(len(sum(sendingProd,[]))+1)
    myCalMat = np.append(
        dummy,
        allFoodTransported-np.sum(dummy)).reshape((len(senderIDs),
        len(receiverIDs))
    )

    print(go.calibration(allFlows,myCalMat))




    ===========================================================================
    """
    outputFlow, inputFlow, products, members = flowTensor.shape
    outputCal, inputCal = np.shape(calibrationMatrix)
    if outputFlow != outputCal or inputFlow != inputCal:
        raise Exception(
            "Error: Expected dimensions of the flow matrices and "
            "the calibration measure to be identical. "
            "Please consider using the function 'convertMatrix' "
            "from within this library."
        )

    # start with everything the smallest transport distance
    currentConf = products * [0]

    # initialising the sum of all flows to compare to calibration matrix with
    # the lowest transport distance
    sumFlow = np.add.reduce(
        flowTensor[0:outputFlow, 0:inputFlow, 0:products, 0], axis=2
    )
    run = -1
    for p in range(products):
        for td in range(members):
            run += 1
            functionalOLD = np.abs(sumFlow - calibrationMatrix).sum(1).sum(0)

            # changing the ensemble product by product
            whichP = np.mod(run, products)
            oldFlow = flowTensor[0:outputFlow, 0:inputFlow, p, currentConf[p]]

            newFlow = flowTensor[0:outputFlow, 0:inputFlow, p, td]

            # replace old flow with new flow
            newSumFlow = sumFlow - oldFlow + newFlow

            functionalNEW = np.abs(newSumFlow - calibrationMatrix).sum(1).sum(0)

            if functionalNEW < functionalOLD:
                currentConf[p] = td
    print(
        "exiting with relative difference to calibration matrix: "
        + str(
            abs(np.sum(newSumFlow) - np.sum(calibrationMatrix))
            / np.sum(calibrationMatrix)
        )
    )
    return currentConf


def convertMatrix(
    haveSenderAttrs: list,
    haveReceiverAttrs: list,
    inputMatrix: np.array,
    wantSenderAttrs: list,
    wantReceiverAttrs: list,
):
    """
    purpose: converts the input matrix to a contracted version or a widened
    version of itself
    m = dim(haveSenderAttrs), n = dim(haveReceiverAttrs)
    dim(inputMatrix) = mXn
    p = dim(wantSenderAttrs), q = dim(wantReceiverAttrs)
    An mXn matrix is converted into a pXq matrix.
    If p<m and q<n, the contracted rows and columns are added to each other,
    respectively: Application for calibrationProcess.

    If p>m and q>n, the new rows and columns are copied their contents,
    respectively: Application for quick creation of a distance matrix with
    different actors, sending or receiving.

    input:
    :param haveSenderAttrs: the list of senders connected to the flow matrix
    :param haveReceiverAttrs: the list of receivers connected to the flow
    matrix
    :param inputMatrix: the matrix to either be contracted or extended.
    :param wantSenderAttrs: a list of the attributes of the senders
    :param wantReceiverAttrs: a list of the attributes of the receivers

    :return the converted matrix

    example (for both application types, respectively):
    from supplychainmodelhelper import graphoperations as go
    import pandas as pd

    # creating the graph
    prod = ['milk', 'beer', 'schnaps']
    act = ['producer', 'consumer', 'warehouse', 'store']
    loc = ['BER', 'SXF', 'TXL']
    myNW = go.createGraph(
            listOfActors=act,
            listOfLocations=loc,
            listOfProducts=prod
    )

    # creating list of node IDs of participants
    ids_producer_milk = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['milk']
    )
    ids_warehouse_milk = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['warehouse'],
        products=['milk']
    )
    ids_consumer_milk = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['consumer'],
        products=['milk']
    )
    ids_producer_beer = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['beer']
    )
    ids_consumer_beer = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['consumer'],
        products=['beer']
    )



    # Creating Distance matrix
    myData = {
        loc[0]: [1, 2, 50],
        loc[1]: [2, 1, 49],
        loc[2]: [50, 49, 1]
    }
    myDF = pd.DataFrame(myData, index=loc)


    # get the content 'location' of the participating nodes
    _, location_producer_milk = go.convertLoT2NTs(
        go.getNodeIDswAttr(
            SCGraph=myNW,
            nameOfAttr='location',
            listOfNodeIDs=ids_producer_milk
        )
    )
    _, location_consumer_milk = go.convertLoT2NTs(
        go.getNodeIDswAttr(
            SCGraph=myNW,
            nameOfAttr='location',
            listOfNodeIDs=ids_consumer_milk
        )
    )
    _, location_producer_beer = go.convertLoT2NTs(
        go.getNodeIDswAttr(
            SCGraph=myNW,
            nameOfAttr='location',
            listOfNodeIDs=ids_producer_beer
        )
    )
    _, location_consumer_beer = go.convertLoT2NTs(
        go.getNodeIDswAttr(
            SCGraph=myNW,
            nameOfAttr='location',
            listOfNodeIDs=ids_consumer_beer
        )
    )

    # EXAMPLE 1: expansion
    myTrafo = go.convertMatrix(
        haveSenderAttrs=loc,
        haveReceiverAttrs=loc,
        inputMatrix=myDF.to_numpy(),
        wantSenderAttrs=location_producer_milk+location_producer_beer,
        wantReceiverAttrs=location_consumer_milk+location_consumer_beer+ids_warehouse_milk
    )

    # creating edgeIDs for adding 'distance' to graph
    # needed, because any edge involved in the gravity model needs a distance
    # property
    allcombIDs1, allcombIDs2 = go.getAllCombinations(
        list1=ids_producer_milk+ids_producer_beer,
        list2=ids_consumer_milk+ids_consumer_beer+ids_warehouse_milk,
        order='1st'
    )
    myEdges4Graph = go.getEdgeID(
        SCGraph=myNW,
        outgoingNodes=allcombIDs1,
        incomingNodes=allcombIDs2
    )
    # adding the distance information to all involved edges simultaneously
    didItWork3 = go.addAttr2Edges(
        SCGraph=myNW,
        listOfEdgeIDs=myEdges4Graph,
        listOfContent=myTrafo.flatten().tolist(),
        attr='distance'
    )

    ###############################################################################
    # EXAMPLE 2 (contraction)
    # creating list of node IDs of participants
    senderIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['producer'],
        products=['milk']
    )
    receiverIDs = go.getListOfNodeIDs(
        SCGraph=myNW,
        actors=['consumer', 'warehouse'],
        products=['milk']
    )

    # supply and demand
    sendingMilk = [10, 50, 40]
    receivingMilk = [10, 15, 20, 15, 15, 25]
    go.addAttr2ExistingNodes(myNW, senderIDs, 'output', sendingMilk)
    go.addAttr2ExistingNodes(myNW, receiverIDs, 'input', receivingMilk)

    # get the transport distances
    myTDs = go.calcPotTransportDistances(
        SCGraph=myNW,
        listOfSenderIDs=senderIDs,
        listOfReceiverIDs=receiverIDs
    )

    # Run the gravity model with given transport distance and return the flow
    flow = go.hymanModel(
        SCGraph=myNW,
        listOfSenderIDs=senderIDs,
        listOfReceiverIDs=receiverIDs,
        transportDistance=myTDs[0],
        tolerance=0.01
    )


    locFlowOnly = go.convertMatrix(
        haveSenderAttrs=senderIDs,
        haveReceiverAttrs=receiverIDs,
        inputMatrix=flow,
        wantSenderAttrs=loc,
        wantReceiverAttrs=loc
    )



    ===========================================================================
    """
    p = len(wantSenderAttrs)
    q = len(wantReceiverAttrs)

    m = len(haveSenderAttrs)
    n = len(haveReceiverAttrs)
    if p < 1:
        raise Exception("Error: please check parameter: wantSenderAttrs")
    if q < 1:
        raise Exception("Error: please check parameter: wantReceiverAttrs")
    if m < 1:
        raise Exception("Error: please check parameter: haveSenderAttrs")
    if n < 1:
        raise Exception("Error: please check parameter: haveReceiverAttrs")
    if p == m and q == n:
        raise Exception(
            "Error, dimensions identical. " "Please have a look at the documentation"
        )

    # dim matrices
    # mXn -> pXq
    # pXm * mXn * nXq = pXq

    myListS = []
    myListR = []
    for attr in wantSenderAttrs:
        myListS.append([1 if i == attr else 0 for i in haveSenderAttrs])

    for attr in wantReceiverAttrs:
        myListR.append([1 if i == attr else 0 for i in haveReceiverAttrs])

    # flow matrix is of dimension mXn
    # contracted matrix is of dimension pXp
    # contraction sender matrix (from left) needs to be of dimension pXm
    # contraction receiver matrix (from right) needs to be of dimension nXp
    contMatS = np.array(myListS)
    contMatR = np.array(myListR).transpose()

    return np.dot(np.dot(contMatS, inputMatrix), contMatR)

