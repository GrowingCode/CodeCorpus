//文件名：statelessTrade.java
//远程接口
package stateless;
import javax.ejb.*;
import java.util.*;
import java.rmi.*;
/*
*这是StatefulTrade的远程接口定义。远程接口定义了客户端能远程调用的EJB的方法。
*这些方法除了抛出异常java.rmi.RemoteException，和EJB中定义是一致的。
*但不是EJB来实现这个接口，而是由容器字典产生的类StatefulTradeBeanE实现。
*/
public interface statelessTrade extends javax.ejb.EJBObject {
/*
*方法说明：添加资金
* @参数：fund 资金数
* @返回：
* @异常：RemoteException 当系统通信发生故障时
*/
  public void addFunds(double fund) throws Exception, RemoteException;
/*
*方法说明：提取资金
* @参数：fund 资金数
* @返回：
* @异常：RemoteException 当系统通信发生故障时
*/
  public void removeFunds(double fund) throws Exception, RemoteException;
/*
*方法说明：察看资金数目
* @参数：
* @返回：double 资金数
* @异常：RemoteException 当系统通信发生故障时
*/
  public double getBalance() throws RemoteException;
}
