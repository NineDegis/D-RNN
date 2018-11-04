const convertQuery = (queryResult, attrList) => {
  const numAttrs = attrList.length;
  const convertedQuery = {};
  for (let i = 0; i < numAttrs; i++) {
    convertedQuery[attrList[i]] = queryResult[attrList[i]];
  }
  return convertedQuery;
};

export { convertQuery };
