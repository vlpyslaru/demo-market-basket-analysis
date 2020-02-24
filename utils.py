import numpy
import pandas

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules



def read_orders(path='data/orders.csv.gz'):
    return pandas.read_csv(path, parse_dates=[ 'order_date' ])



def clean_orders(orders):
    orders = orders[orders.inv_qty > 0]
    
    orders = orders.groupby('order_id').filter(lambda group: len(set(group.order_date)) == 1)

    return orders



def encode_orders_materials(orders):
    orders_grouped = orders[[ 'order_id', 'material' ]].groupby('order_id')

    orders_materials = [ list(orders_group.material) for (_, orders_group) in orders_grouped ]

    encoder = TransactionEncoder()
    orders_materials = encoder.fit_transform(orders_materials, sparse=True)
    
    orders_index = list(orders_grouped.groups.keys())
    orders_columns = [ str(column) for column in encoder.columns_ ]

    return pandas.DataFrame.sparse.from_spmatrix(orders_materials, index=orders_index, columns=orders_columns)



def encode_orders(orders):
    orders_grouped = orders[[ 'order_id', 'order_date', 'org' ]].groupby('order_id').first()

    orders_materials = encode_orders_materials(orders)

    return pandas.concat([ orders_grouped, orders_materials ], axis='columns', join='inner')



def locate_orders_encoded(orders, indexes):
    orders_dense = orders[[ 'order_date', 'org' ]].loc[indexes]

    columns_sparse = set(orders.columns).difference(set({ 'order_date', 'org' }))

    orders_sparse = orders[columns_sparse].sparse.to_coo().tocsr()
    orders_sparse = orders_sparse[orders.index.isin(indexes)]
    
    index_sparse = orders.index[orders.index.isin(indexes)]
    orders_sparse = pandas.DataFrame.sparse.from_spmatrix(orders_sparse, index=index_sparse, columns=columns_sparse)

    return pandas.concat([ orders_dense, orders_sparse ], axis='columns', join='inner')



def yield_materials_support(orders, threshold=None):
    for material in set(orders.columns).difference(set({ 'order_date', 'org' })):
        series = orders[material]

        assert series.dtype == pandas.SparseDtype(bool)

        support = None

        if not series.sparse.fill_value:
            support = series.sparse.density
        else:
            support = 1 - series.sparse.density

        if (threshold is None) or (support >= threshold):
            yield (material, support)



def mine_associations(orders, support_threshold=None, confidence_threshold=None, lift_threshold=None, assert_materials_count=1000):
    materials = dict(yield_materials_support(orders, support_threshold))

    assert len(materials) <= assert_materials_count, 'Supported materials count above {} might have drastic impact on performance'.format(assert_materials_count)

    X = orders[list(materials.keys())]

    if support_threshold is None:
        support_threshold = 0

    if confidence_threshold is None:
        confidence_threshold = 0

    y = fpgrowth(X, min_support=support_threshold, use_colnames=True)
    y = association_rules(y, min_threshold=confidence_threshold)
    
    if lift_threshold is not None:
        y = y[y['lift'] >= lift_threshold]

    return y


