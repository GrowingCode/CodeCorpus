/**
 *second 
 */
package com.mingrisoft.DOM_demo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * @author baiweiming
 * 
 */
public class ElementNameDOMParserFile {

	private List<String> elementList = new ArrayList<String>();

	/**
	 * 读取XML文件
	 * 
	 * @param path
	 * @return
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public Document parseReadFile(String path)
			throws ParserConfigurationException, SAXException, IOException {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = documentBuilderFactory
				.newDocumentBuilder();
		File file = new File(path);
		return dombuilder.parse(file);
	}

	/**
	 * 获取{@link Node}的name
	 * 
	 * @param parentNode
	 * @return
	 */
	public List<String> getElementName(Node parentNode) {
		if (parentNode.hasChildNodes()) {
			NodeList nodeList = parentNode.getChildNodes();
			for (int i = 0; i < nodeList.getLength(); i++) {
				Node node = nodeList.item(i);
				if (node.hasChildNodes()) {
					getElementName(node);
					elementList.add((node.getNodeName()));
				}
					
			}
		}
		return elementList;
	}

	public static void main(String[] arg) {
		ElementNameDOMParserFile parserFile = new ElementNameDOMParserFile();
		String path = "xmldemo/books.xml";
		try {

			Document document = parserFile.parseReadFile(path);
			List<String> list = parserFile.getElementName(document);
			System.out.println("XML元素名称");
			System.out.println(list);
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
