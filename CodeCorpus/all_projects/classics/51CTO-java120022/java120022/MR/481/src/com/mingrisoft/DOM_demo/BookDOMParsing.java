package com.mingrisoft.DOM_demo;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.DOMException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import com.mingrisoft.entity.Book;
import com.mingrisoft.entity.BookPrice;

/**
 * @author bwm
 * 
 */
public class BookDOMParsing {

	private List<Book> bookList = new ArrayList<Book>();

	private Book book;

	/**
	 * 读取XML文件
	 * 
	 * @param path
	 * @return
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	public Document parseReadFile(String path) {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = null;
		try {
			dombuilder = documentBuilderFactory.newDocumentBuilder();
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		File file = new File(path);
		try {
			return dombuilder.parse(file);
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * 获取{@link Node}的name和textcontent
	 * 
	 * @param document
	 */
	public List<Book> getBook(Element element) {
		NodeList list = element.getElementsByTagName("book:book");

		for (int i = 0; i < list.getLength(); i++) {
			book = new Book();
			NodeList name = element.getElementsByTagName("book:name");
			NodeList publisher = element.getElementsByTagName("book:publisher");
			NodeList company = element.getElementsByTagName("book:company");
			NodeList author = element.getElementsByTagName("book:author");
			NodeList ISBN = element.getElementsByTagName("book:ISBN");
			NodeList price = element.getElementsByTagName("book:price");
			NodeList url = element.getElementsByTagName("book:url");
			book.setName(name.item(i).getTextContent());
			book.setPublisher(publisher.item(i).getTextContent());
			book.setCompany(company.item(i).getTextContent());
			book.setAuthor(author.item(i).getTextContent());
			book.setIsbn(ISBN.item(i).getTextContent());

			// 获取价格
			BookPrice bookPrice = new BookPrice();
			Map<String, String> attributeMap = new HashMap<String, String>();
			NamedNodeMap namedNodeMap = price.item(i).getAttributes();
			for (int j = 0; j < namedNodeMap.getLength(); j++) {
				// 获取价格属性
				Node node = namedNodeMap.item(j);
				attributeMap.put(node.getNodeName(), node.getNodeValue());
			}
			bookPrice.setAttributeMap(attributeMap);
			bookPrice.setValue(new Double(price.item(i).getTextContent()));
			book.setPrice(bookPrice);

			// 获取URL
			try {
				book.setUrl(new URL(url.item(i).getTextContent()));
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (DOMException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			bookList.add(book);
		}
		return bookList;
	}

	public static void main(String[] arg) {

		String pathname = "xmldemo/books.xml";
		BookDOMParsing elementSAXParsing = new BookDOMParsing();
		Document document = null;
		document = elementSAXParsing.parseReadFile(pathname);
		List<Book> bookElements = elementSAXParsing.getBook(document
				.getDocumentElement());
		System.out.println("自定义JAVA类封装元素名称和内容");
		System.out.println(bookElements);

	}
}
