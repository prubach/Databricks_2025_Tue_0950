# Databricks notebook source
# MAGIC %sh
# MAGIC ls -al /dbfs/FileStore/tables/users.txt

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.graphx.GraphLoader
# MAGIC
# MAGIC // Load the edges as a graph
# MAGIC //file_location = "/FileStore/tables/users.txt"
# MAGIC val graph = GraphLoader.edgeListFile(sc, "/FileStore/tables/followers-2.txt")
# MAGIC
# MAGIC //println(graph.triangleCount())
# MAGIC
# MAGIC println(graph.triangleCount().vertices.collect().mkString("\n"))

# COMMAND ----------



# COMMAND ----------

# MAGIC %scala
# MAGIC
# MAGIC import org.apache.spark.graphx.GraphLoader
# MAGIC
# MAGIC // Load the edges as a graph
# MAGIC //file_location = "/FileStore/tables/users.txt"
# MAGIC val graph = GraphLoader.edgeListFile(sc, "/FileStore/tables/followers.txt")
# MAGIC // Run PageRank
# MAGIC val ranks = graph.pageRank(0.1).vertices
# MAGIC
# MAGIC println(ranks.collect().mkString("\n"))
# MAGIC // Join the ranks with the usernames
# MAGIC val users = sc.textFile("/FileStore/tables/users.txt").map { line =>
# MAGIC   val fields = line.split(",")
# MAGIC   (fields(0).toLong, fields(1))
# MAGIC }
# MAGIC val ranksByUsername = users.join(ranks).map {
# MAGIC   case (id, (username, rank)) => (username, rank)
# MAGIC }
# MAGIC // Print the result
# MAGIC println(ranksByUsername.collect().mkString("\n"))
# MAGIC

# COMMAND ----------

# File location and type
file_location = '/FileStore/tables/users.txt'
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_users = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
df_users = df_users.withColumnRenamed('_c0','id')
df_users = df_users.withColumnRenamed('_c1','login')
df_users = df_users.withColumnRenamed('_c2','name')

display(df_users)

# COMMAND ----------

# File location and type
#file_location = '/FileStore/tables/followers-2.txt'
file_location = '/FileStore/tables/followers2-1.txt'
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "false"
delimiter = " "

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

df = df.withColumnRenamed('_c0','src')
df = df.withColumnRenamed('_c1','dst')

display(df_users)
display(df)

# COMMAND ----------

import networkx as nx
import graphframes
g = graphframes.GraphFrame(df_users, df)

gp = nx.from_pandas_edgelist(df.toPandas(),'src','dst')
nx.draw(gp, with_labels = True, arrows=True)

# COMMAND ----------

pos = nx.spring_layout(gp)
nx.draw(gp, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')

# COMMAND ----------

G = nx.DiGraph() #or G = nx.MultiDiGraph()
G.add_node('1')
G.add_node('2')
G.add_node('3')
G.add_node('4')
G.add_node('5')
G.add_edge('1', '5')
G.add_edge('1', '3')
G.add_edge('2', '1')
G.add_edge('2', '5')
G.add_edge('2', '3')
G.add_edge('2', '4')
G.add_edge('4', '3')
G.add_edge('3', '2')
G.add_edge('5', '3')
G.add_edge('5', '2')

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')



# COMMAND ----------

import networkx as nx
from graphframes import GraphFrame
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (5,5)

def PlotGraph(edge_list):
    Gplot=nx.DiGraph()
    for row in edge_list.select('src','dst').take(100):
        Gplot.add_edge(row['src'],row['dst'])
    nx.draw_networkx(Gplot, arrows=True, with_labels=True, width=1, arrowstyle = '-|>', arrowsize = 12, node_color = 'yellow', node_size = 1000)

 

PlotGraph(g.edges)

# COMMAND ----------

from graphviz import Digraph

dot = Digraph()
dot.node('A', 'A')
dot.node('B', 'B')
dot.node('C', 'C')
dot.edges(['AB', 'AB', 'AB', 'BC', 'BA', 'CB'])

print(dot.source)
dot.render('/tmp/outgraph', format='svg', view=True)



# COMMAND ----------

print(g)
# Query: Get in-degree of each vertex.
g.inDegrees.show()
g.outDegrees.show()

# Query: Count the number of "follow" connections in the graph.
#num_followers = g.edges.filter("relationship = 'follow'").count()

#print(num_followers)
# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.2, maxIter=10)
#
results = g.pageRank(resetProbability=0.2, tol=0.05)
print(results)
results.vertices.select("id", "pagerank").show()

# COMMAND ----------

results = g.parallelPersonalizedPageRank(resetProbability=0.2, sourceIds=['1'], maxIter=10)
print(results)
#results.vertices.select("id", "login", "").show()

# COMMAND ----------

g.vertices.show()

# COMMAND ----------

paths = g.bfs("id==7", "id==2", maxPathLength=5)
paths.show()


# COMMAND ----------


#from graphframes import *
import graphframes


# Create a Vertex DataFrame with unique ID column "id"
vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])


# Create an Edge DataFrame with "src" and "dst" columns
edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
  ("f", "c", "follow"),
  ("e", "f", "follow"),
  ("e", "d", "friend"),
  ("d", "a", "friend"),
  ("a", "e", "friend")
], ["src", "dst", "relationship"])

print(vertices)
print(edges)

# Create a GraphFrame
g = graphframes.GraphFrame(vertices, edges)
print(g)

# Query: Get in-degree of each vertex.
g.inDegrees.show()

# Query: Count the number of "follow" connections in the graph.
num_followers = g.edges.filter("relationship = 'follow'").count()

print(num_followers)
# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.2, maxIter=10)
#
results = g.pageRank(resetProbability=0.2, tol=0.05)
print(results)
#results.vertices.select("id", "pagerank").show()

# COMMAND ----------

results.vertices.select("id", "pagerank").show()

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.triangleCount())

# COMMAND ----------


from graphframes import *

# Create a Vertex DataFrame with unique ID column "id"
vertices = sqlContext.createDataFrame([
  ("ID1",),
  ("ID2",),
  ("ID3",),
  ("ID4",),
  ("ID5",) ], ["id"])

# Create an Edge DataFrame with "src" and "dst" columns
edges = sqlContext.createDataFrame([
  ("ID1", "ID3"),
  ("ID1", "ID5"),
  ("ID2", "ID1"),
  ("ID2", "ID3"),
  ("ID2", "ID4"),
  ("ID2", "ID5"),
  ("ID3", "ID2"),
  ("ID4", "ID3"),
  ("ID5", "ID2"),
  ("ID5", "ID3"),
], ["src", "dst"])

print(g)

vertices.cache()
edges.cache()

#

# Query: Get in-degree of each vertex.
g.inDegrees.show()

# Query: Count the number of "follow" connections in the graph.
#g.edges.filter("relationship = 'follow'").count()

# Run PageRank algorithm, and show results.
#results = g.pageRank(resetProbability=0.2, maxIter=5, tol=0.1)
results = g.pageRank(resetProbability=0.15, tol=0.1)
#print(results)
results.vertices.select("id", "pagerank").show()

# COMMAND ----------

display(g.vertices)

# COMMAND ----------

g2 = GraphFrame(vertices, edges)
print(g2)
results = g2.parallelPersonalizedPageRank(resetProbability=0.2, sourceIds=['ID1'], maxIter=10)
#print(results)
results.vertices.select("id", "pagerank").show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.graphx.{GraphLoader, PartitionStrategy, EdgeDirection, VertexRDD, VertexId, Graph}
# MAGIC
# MAGIC // Load the edges in canonical order and partition the graph for triangle count
# MAGIC val graph = GraphLoader.edgeListFile(sc, "/FileStore/tables/followers2.txt", true)
# MAGIC   .partitionBy(PartitionStrategy.RandomVertexCut)
# MAGIC
# MAGIC val verticesWithSuccessors: VertexRDD[Array[VertexId]] = 
# MAGIC     graph.ops.collectNeighborIds(EdgeDirection.Out)
# MAGIC
# MAGIC val successorGraph = Graph(verticesWithSuccessors, graph.edges)
# MAGIC val adjList = successorGraph.vertices
# MAGIC
# MAGIC val df = adjList.toDF(Seq("node", "adjacents"): _*)
# MAGIC //println(df)
# MAGIC df.show()
# MAGIC
# MAGIC val nodsAdjacents = users.join(df).map { case (id, (node, adjacents)) =>
# MAGIC   (id, node, adjacents)
# MAGIC }
# MAGIC println(nodsAdjacents.collect().mkString("\n"))
# MAGIC
# MAGIC
# MAGIC // Find the triangle count for each vertex
# MAGIC val triCounts = graph.triangleCount().vertices
# MAGIC // Join the triangle counts with the usernames
# MAGIC val users = sc.textFile("/FileStore/tables/users.txt").map { line =>
# MAGIC   val fields = line.split(",")
# MAGIC   (fields(0).toLong, fields(1))
# MAGIC }
# MAGIC
# MAGIC
# MAGIC
# MAGIC val triCountByUsername = users.join(triCounts).map { case (id, (username, tc)) =>
# MAGIC   (username, tc)
# MAGIC }
# MAGIC // Print the result
# MAGIC println(triCountByUsername.collect().mkString("\n"))
# MAGIC
# MAGIC //println(graph.triangleCount().vertices)
